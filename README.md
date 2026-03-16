# RLSnake

## Introduction
A Snake game with a Deep Q-Network (DQN) agent that learns to play through reinforcement learning. The project was built to explore reinforcement learning, and see the approach needed for it to tackle a more difficult problem than what was studied at university. Simple Q-learning does not work with Snake, so I had to experiment with DQN to make it work at all.


## Demo

![Trained AI Agent](assets/trained_agent.gif)


## How it works

The agent observes an 11-feature state vector each step:

- Whether the next cell is dangerous in 3 relative directions (straight, right, left)
- Which cardinal direction the snake is currently moving
- Which direction the food is relative to the head

It outputs one of 3 actions — go straight, turn right, or turn left. A feedforward neural network maps state to Q-values (estimated future reward per action), trained via the Bellman equation using experience replay.

## Requirements

- Python 3.13 (specified in .python-version)
- [uv](https://github.com/astral-sh/uv) (highly recommended)

Install dependencies:

```bash
uv sync
```

Or with pip:

```bash
pip install pygame torch numpy
```


## Running

```bash
python main.py
```

| Key | Action |
|-----|--------|
| ↑ / ↓ | Increase / decrease speed (±5) |
| F | Toggle rendering on/off for faster training |
| Space | Pause |

The model is saved to `model.pth` when you close the window and loaded automatically on the next run.


## Project structure

```
main.py      Entry point and training loop
board.py     Game logic and environment (state, step, render)
agent.py     DQN agent (epsilon-greedy, experience replay, Bellman update)
model.py     Neural network definition and save/load
model.pth    Saved weights (created after first training session)
```


## Things to improve

- **Give the agent a full board view** — currently it only sees 11 signals about its immediate surroundings. Feeding it the entire grid as an image (using a CNN) would let it actually see where its body is and plan ahead rather than reacting one step at a time.

- **Self-trapping** — the agent still boxes itself in at higher scores because it has no way to tell that a sequence of moves will cut off its own escape route. Flood fill features were tested to address this but made performance worse, so this remains an open problem with the current approach.

- **Target network** — standard DQN uses a second copy of the network that updates slowly, which stabilises training. Right now the agent trains against its own live predictions, which can cause it to chase a moving target and learn inconsistently.

- **Train on a small board first** — starting on a 10x10 board and moving to 20x20 once the agent is competent would speed up early training significantly, since games end faster and patterns are simpler to learn.

- **Save training metrics** — scores and game count are lost when the window closes. Logging them to a file would make it easy to plot the learning curve and compare different runs if needed.
