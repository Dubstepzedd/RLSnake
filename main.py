"""
RLSnake  –  DQN agent training
--------------------------------
Run:   python main.py

Controls:  ↑/↓ = faster/slower  F = toggle render  SPACE = pause
"""
import pygame

from board import Board, HEADER_HEIGHT
from agent import Agent


def run_ai() -> None:
    """Initialise pygame, create the board and agent, then run the training loop.

    Handles keyboard input for speed control, render toggle, and pause.
    Saves the model to disk when the window is closed.
    """
    pygame.init()
    pygame.display.set_caption("RLSnake")

    board  = Board()
    screen = pygame.display.set_mode((board.width, board.height))
    pygame.key.set_repeat(200, 80)
    clock  = pygame.time.Clock()

    agent = Agent()

    scores    : list[int] = []
    speed     : int   = 1
    render    : bool  = True
    paused    : bool  = False
    move_timer: float = 0.0

    font_big = pygame.font.SysFont("consolas", 36, bold=True)

    running = True
    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    speed = min(speed + 5, 30)
                if event.key == pygame.K_DOWN:
                    speed = max(speed - 5, 1)
                if event.key == pygame.K_f:
                    render = not render
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if paused:
            if render:
                _draw_pause(screen, board, font_big)
                pygame.display.flip()
            continue

        # speed 1 = one move per 150ms; higher speeds scale the interval down
        move_interval = 150.0 / speed
        move_timer += dt
        steps = 0
        while move_timer >= move_interval:
            move_timer -= move_interval
            steps += 1

        for _ in range(steps):
            state  = board.get_state()
            action = agent.get_action(state)
            reward = board.step(action)
            alive  = board.is_alive()
            next_state = board.get_state()

            agent.remember(state, action, reward, next_state, not alive)
            agent.train_short_memory(state, action, reward, next_state, not alive)

            if not alive:
                agent.n_games += 1
                agent.epsilon = max(0.01, agent.epsilon - 0.005)
                agent.train_long_memory()
                scores.append(board.score)
                board.games_played = agent.n_games
                board.reset()
                break

        if render:
            avg50 = sum(scores[-50:]) / max(1, len(scores[-50:])) if scores else 0.0
            extra = {
                "Games": agent.n_games,
                "\u03b5": f"{agent.epsilon:.2f}",
                "Avg50": f"{avg50:.1f}",
                "Speed": f"x{speed}",
            }
            board.render(screen, extra_stats=extra)
            pygame.display.flip()

    agent.save()
    pygame.quit()


def _draw_pause(screen: pygame.Surface, board: Board, font: pygame.font.Font) -> None:
    """Render the current board with a semi-transparent pause overlay.

    Args:
        screen: The pygame surface to draw onto.
        board:  The board instance used for rendering.
        font:   Font used to render the "PAUSED" label.
    """
    board.render(screen)
    ov = pygame.Surface((board.width, board.rows * board.cell_size), pygame.SRCALPHA)
    ov.fill((0, 0, 0, 140))
    screen.blit(ov, (0, HEADER_HEIGHT))
    text = font.render("PAUSED", True, (255, 210, 60))
    cx   = board.width // 2
    cy   = HEADER_HEIGHT + board.rows * board.cell_size // 2
    screen.blit(text, text.get_rect(center=(cx, cy)))


if __name__ == "__main__":
    run_ai()
