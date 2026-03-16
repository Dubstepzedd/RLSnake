import random
from typing import Optional

import pygame

# Colors
BG_COLOR      = (15, 17, 26)
GRID_COLOR    = (30, 33, 48)
HEAD_COLOR    = (80, 220, 100)
BODY_COLOR    = (45, 155, 65)
APPLE_COLOR   = (220, 55, 55)
APPLE_SHINE   = (255, 130, 130)
TEXT_COLOR    = (190, 195, 215)
HEADER_BG     = (10, 12, 20)
SCORE_COLOR   = (255, 210, 60)
BORDER_COLOR  = (50, 55, 80)

CELL_SIZE     = 30
HEADER_HEIGHT = 55
COLS          = 20
ROWS          = 20

type Position  = tuple[int, int]
type Direction = tuple[int, int]


class Board:
    def __init__(self, cols: int = COLS, rows: int = ROWS) -> None:
        """Initialise the board, fonts, and persistent stats, then reset game state.

        Args:
            cols: Number of columns in the grid.
            rows: Number of rows in the grid.
        """
        self.cols      : int = cols
        self.rows      : int = rows
        self.cell_size : int = CELL_SIZE
        self.width     : int = cols * CELL_SIZE
        self.height    : int = rows * CELL_SIZE + HEADER_HEIGHT

        self.score       : int = 0
        self.high_score  : int = 0
        self.games_played: int = 0

        self._font      : Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None

        self.reset()

    def reset(self) -> None:
        """Reset game state for a new episode. Persistent stats (high_score, games_played) are kept."""
        mid_col = self.cols // 2
        mid_row = self.rows // 2
        self.snake         : list[Position] = [(mid_col, mid_row), (mid_col - 1, mid_row)]
        self.direction     : Direction      = (1, 0)
        self.next_direction: Direction      = (1, 0)
        self.apple         : Position       = (0, 0)
        self.score               : int  = 0
        self.alive               : bool = True
        self.ate_apple           : bool = False
        self.steps_without_apple : int  = 0
        self.max_steps           : int  = self.cols * self.rows * 2
        self._place_apple()

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _place_apple(self) -> None:
        """Randomly place the apple on an unoccupied cell."""
        while True:
            apple: Position = (random.randint(0, self.cols - 1), random.randint(0, self.rows - 1))
            if apple not in self.snake:
                self.apple = apple
                break

    def _move(self) -> None:
        """Advance the snake one step in the current direction.

        Sets self.alive to False on wall collision, self collision, or step limit exceeded.
        Sets self.ate_apple to True if the new head lands on the apple.
        """
        head: Position     = self.snake[0]
        new_head: Position = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Wall collision
        if new_head[0] < 0 or new_head[0] >= self.cols or new_head[1] < 0 or new_head[1] >= self.rows:
            self.alive     = False
            self.ate_apple = False
            return

        # Self collision
        if new_head in self.snake[1:]:
            self.alive     = False
            self.ate_apple = False
            return

        self.steps_without_apple += 1
        if self.steps_without_apple > self.max_steps:
            self.alive     = False
            self.ate_apple = False
            return

        self.ate_apple = (new_head == self.apple)
        if self.ate_apple:
            self.snake.insert(0, new_head)
            self.score += 1
            self.high_score = max(self.high_score, self.score)
            self.steps_without_apple = 0
            self._place_apple()
        else:
            self.snake = [new_head] + self.snake[:-1]

    def _dist_to_apple(self) -> int:
        """Return the Manhattan distance from the snake's head to the apple.

        Returns:
            Manhattan distance as a non-negative integer.
        """
        head = self.snake[0]
        return abs(head[0] - self.apple[0]) + abs(head[1] - self.apple[1])

    # ------------------------------------------------------------------ #
    #  RL interface                                                        #
    # ------------------------------------------------------------------ #

    def get_state(self) -> list[int | float]:
        """Return the 11-feature state vector observed by the RL agent.

        Returns:
            List of 11 binary integers:
            [0-2]  Danger in the straight, right, and left directions.
            [3-6]  Current movement direction as one-hot (left, right, up, down).
            [7-10] Food position relative to head as one-hot (left, right, up, down).
        """
        head: Position    = self.snake[0]
        dx, dy = self.direction

        right_dir: Direction = (-dy, dx)
        left_dir : Direction = (dy, -dx)

        def is_danger(d: Direction) -> bool:
            x, y = head[0] + d[0], head[1] + d[1]
            if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
                return True
            return (x, y) in self.snake[1:-1]  # tail moves away, not a real obstacle

        return [
            # Danger: straight, right, left
            int(is_danger(self.direction)),
            int(is_danger(right_dir)),
            int(is_danger(left_dir)),
            # Current direction
            int(dx == -1),  # left
            int(dx ==  1),  # right
            int(dy == -1),  # up
            int(dy ==  1),  # down
            # Food relative position
            int(self.apple[0] < head[0]),  # food left
            int(self.apple[0] > head[0]),  # food right
            int(self.apple[1] < head[1]),  # food up
            int(self.apple[1] > head[1]),  # food down
        ]

    def set_action(self, action: int) -> None:
        """Convert a relative action into an absolute next direction.

        Args:
            action: 0 = go straight, 1 = turn right, 2 = turn left.
        """
        dx, dy = self.direction
        if action == 1:
            self.next_direction = (-dy, dx)
        elif action == 2:
            self.next_direction = (dy, -dx)
        else:
            self.next_direction = self.direction

    def step(self, action: int) -> float:
        """Perform one RL step and return the reward.

        Args:
            action: 0 = straight, 1 = turn right, 2 = turn left.

        Returns:
            +10.0 for eating an apple, -10.0 for dying,
            +0.1 / -0.1 for moving closer / further from the apple.
        """
        self.set_action(action)
        self.direction = self.next_direction

        prev_dist = self._dist_to_apple()
        self._move()

        if not self.alive:
            return -10.0
        if self.ate_apple:
            return 10.0

        new_dist = self._dist_to_apple()
        return 0.1 if new_dist < prev_dist else -0.1

    def is_alive(self) -> bool:
        """Return whether the snake is still alive.

        Returns:
            True if the snake has not collided or exceeded the step limit.
        """
        return self.alive

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def _get_fonts(self) -> tuple[pygame.font.Font, pygame.font.Font]:
        """Lazily initialise the pygame fonts on first call.

        Returns:
            Tuple of (font, small_font) for header rendering.
        """
        if self._font is None:
            self._font       = pygame.font.SysFont("consolas", 19, bold=True)
            self._small_font = pygame.font.SysFont("consolas", 14)
        return self._font, self._small_font

    def render(self, screen: pygame.Surface, extra_stats: Optional[dict[str, str]] = None) -> None:
        """Draw the full frame: header, grid, apple, snake, and optional training stats.

        Args:
            screen:      The pygame surface to draw onto.
            extra_stats: Optional key-value pairs displayed in the header (e.g. games, epsilon).
        """
        font, small_font = self._get_fonts()

        # Header
        pygame.draw.rect(screen, HEADER_BG, (0, 0, self.width, HEADER_HEIGHT))
        pygame.draw.line(screen, BORDER_COLOR, (0, HEADER_HEIGHT), (self.width, HEADER_HEIGHT), 2)

        # Score / best in header
        score_surf = font.render(f"SCORE  {self.score:>4}", True, SCORE_COLOR)
        best_surf  = font.render(f"BEST  {self.high_score:>4}", True, TEXT_COLOR)
        screen.blit(score_surf, (12, 10))
        screen.blit(best_surf,  (12, 30))

        # Extra training stats — evenly fill the space right of score/best
        if extra_stats:
            items   = list(extra_stats.items())
            x_start = 185
            gap     = (self.width - x_start) // max(len(items), 1)
            for i, (key, val) in enumerate(items):
                s = small_font.render(f"{key}: {val}", True, TEXT_COLOR)
                screen.blit(s, (x_start + i * gap, 20))

        # Game area background
        game_rect = pygame.Rect(0, HEADER_HEIGHT, self.width, self.rows * self.cell_size)
        pygame.draw.rect(screen, BG_COLOR, game_rect)

        # Grid lines
        for col in range(self.cols + 1):
            x = col * self.cell_size
            pygame.draw.line(screen, GRID_COLOR,
                             (x, HEADER_HEIGHT),
                             (x, HEADER_HEIGHT + self.rows * self.cell_size))
        for row in range(self.rows + 1):
            y = HEADER_HEIGHT + row * self.cell_size
            pygame.draw.line(screen, GRID_COLOR, (0, y), (self.width, y))

        # Apple
        ax = self.apple[0] * self.cell_size + self.cell_size // 2
        ay = HEADER_HEIGHT + self.apple[1] * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 2
        pygame.draw.circle(screen, APPLE_COLOR, (ax, ay), radius)
        pygame.draw.circle(screen, APPLE_SHINE, (ax - radius // 3, ay - radius // 3), max(2, radius // 3))

        # Snake
        for i, seg in enumerate(self.snake):
            x  = seg[0] * self.cell_size + 2
            y  = HEADER_HEIGHT + seg[1] * self.cell_size + 2
            sz = self.cell_size - 4
            color = HEAD_COLOR if i == 0 else BODY_COLOR
            pygame.draw.rect(screen, color, pygame.Rect(x, y, sz, sz), border_radius=5)

        # Eye on head
        if self.snake:
            hx = self.snake[0][0] * self.cell_size
            hy = HEADER_HEIGHT + self.snake[0][1] * self.cell_size
            dx, dy = self.direction
            ex = hx + self.cell_size // 2 + dx * 5
            ey = hy + self.cell_size // 2 + dy * 5
            pygame.draw.circle(screen, (10, 10, 10), (ex, ey), 3)
