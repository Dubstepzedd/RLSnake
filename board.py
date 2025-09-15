import pygame
import random

class Board:

    def __init__(self, px_width, px_height, rows, cols):
        self.width = px_width / cols
        self.height = px_height / rows
        self.rows = rows
        self.cols = cols
        self.apple = (random.randint(0, self.cols - 1), random.randint(0, self.rows - 1))
        self.snake = [(random.randint(0, self.cols - 1), random.randint(0, self.rows - 1))]
        self.direction = (1, 0)
        self.move_timer = 0.0
        self.move_interval = 175.0  # milliseconds between moves

    def update(self, delta_time):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] and self.direction != (0, 1):
            self.direction = (0, -1)
        if keys[pygame.K_a] and self.direction != (1, 0):
            self.direction = (-1, 0)
        if keys[pygame.K_s] and self.direction != (0, -1):
            self.direction = (0, 1)
        if keys[pygame.K_d] and self.direction != (-1, 0):
            self.direction = (1, 0)

        # Accumulate time and only move snake when enough time has passed
        self.move_timer += delta_time
        if self.move_timer >= self.move_interval:
            self.move_timer -= self.move_interval

            new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

            # You might not want this?
            # Wrap around screen
            new_head = (new_head[0] % self.cols, new_head[1] % self.rows)

            if new_head == self.apple:
                self.snake.insert(0, new_head)
                # This works, but could be inefficient if the snake is very long
                while True:
                    new_apple = (random.randint(0, self.cols - 1), random.randint(0, self.rows - 1))
                    if new_apple not in self.snake:
                        self.apple = new_apple
                        break
            else:
                # Move the snake body
                for i in range(len(self.snake) - 1, 0, -1):
                    self.snake[i] = self.snake[i - 1]
                self.snake[0] = new_head

    def is_alive(self):
        head = self.snake[0]
        # Check wall collisions
        if head[0] < 0 or head[0] >= self.cols or head[1] < 0 or head[1] >= self.rows:
            return False
        # Check self collisions
        if head in self.snake[1:]:
            return False
        return True

    def render(self, screen):
        for row in range(self.rows):
            for col in range(self.cols):
                color = "black"
                if (col, row) == self.apple:
                    color = "red"
                rect = pygame.Rect(
                    self.width * col,
                    self.height * row,
                    self.width,
                    self.height
                )

                pygame.draw.rect(screen, color, rect, 1 if color == "black" else 0)


        for segment in self.snake:
            rect = pygame.Rect(
                self.width * segment[0],
                self.height * segment[1],
                self.width,
                self.height
            )

            pygame.draw.rect(screen, "green", rect)
