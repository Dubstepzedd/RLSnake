import random
from collections import deque
from collections.abc import Sequence

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import QNetwork

MAX_MEMORY: int   = 100_000
BATCH_SIZE: int   = 1_000
LR        : float = 0.001
GAMMA     : float = 0.9

type State      = list[int | float]
type Transition = tuple[State, int, float, State, bool]


class Agent:
    def __init__(self) -> None:
        """Initialise the DQN agent, replay buffer, and neural network.

        Loads existing weights from disk if available and sets epsilon low
        so a previously trained agent exploits immediately rather than exploring.
        """
        self.n_games: int   = 0
        self.epsilon: float = 1.0
        self.memory : deque[Transition] = deque(maxlen=MAX_MEMORY)

        self.model    : QNetwork   = QNetwork(11, 256, 3)
        self.optimizer: optim.Adam = optim.Adam(self.model.parameters(), lr=LR)

        if self.model.load():
            self.epsilon = 0.01  # already trained — exploit immediately

    def get_action(self, state: State) -> int:
        """Choose an action using an epsilon-greedy policy.

        Args:
            state: The current 11-feature state vector.

        Returns:
            Action integer: 0 = straight, 1 = turn right, 2 = turn left.
        """
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        with torch.no_grad():
            q = self.model(torch.tensor(state, dtype=torch.float))
        return int(torch.argmax(q).item())

    def remember(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        """Store a transition in the replay buffer.

        Args:
            state:      State before the action.
            action:     Action taken (0, 1, or 2).
            reward:     Reward received after the action.
            next_state: State after the action.
            done:       True if the episode ended on this step.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        """Train the network on a single transition immediately after each step.

        Args:
            state:      State before the action.
            action:     Action taken (0, 1, or 2).
            reward:     Reward received after the action.
            next_state: State after the action.
            done:       True if the episode ended on this step.
        """
        self._train([state], [action], [reward], [next_state], [done])

    def train_long_memory(self) -> None:
        """Train the network on a random batch from the replay buffer.

        Called once at the end of each game. Uses the full buffer if it contains
        fewer than BATCH_SIZE transitions.
        """
        batch: list[Transition] = (random.sample(self.memory, BATCH_SIZE)
                                   if len(self.memory) >= BATCH_SIZE
                                   else list(self.memory))
        states, actions, rewards, next_states, dones = zip(*batch)
        self._train(states, actions, rewards, next_states, dones)

    def save(self) -> None:
        """Persist the current model weights to disk."""
        self.model.save()

    # ------------------------------------------------------------------ #

    def _train(self, states: Sequence[State], actions: Sequence[int], rewards: Sequence[float], next_states: Sequence[State], dones: Sequence[bool]) -> None:
        """Run one gradient update using the Bellman equation.

        Args:
            states:      Batch of states before each action.
            actions:     Batch of actions taken.
            rewards:     Batch of rewards received.
            next_states: Batch of states after each action.
            dones:       Batch of episode-end flags.
        """
        states      = torch.tensor(np.array(states),      dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions     = torch.tensor(actions, dtype=torch.long)
        rewards     = torch.tensor(rewards, dtype=torch.float)
        dones       = torch.tensor(dones,   dtype=torch.bool)

        # Q(s, a)
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ * max Q(s', a')
        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0]
            target_q   = rewards + GAMMA * max_next_q * (~dones)

        # Compute how wrong the network was
        loss = F.mse_loss(current_q, target_q)
        # Reset from the above training step
        self.optimizer.zero_grad()
        # Backpropagation, calculate how much each weight contributed to the loss
        loss.backward()
        # Nudge the weights
        self.optimizer.step()
