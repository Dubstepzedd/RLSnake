# Example file showing a basic pygame "game loop"
import pygame
from board import Board

def init(width, height):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    return screen

def run_game():
    width, height = 500, 500
    screen = init(width, height)
    board = Board(width, height, 10, 10)
    clock = pygame.time.Clock()

    while board.is_alive():
        # Get delta time in milliseconds
        delta_time = clock.tick(60)  # This returns milliseconds since last call
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("white")

        board.update(delta_time)
        board.render(screen)

        # flip() the display to put your work on screen
        pygame.display.flip()

    pygame.quit()
    return len(board.snake)


if __name__ == "__main__":
    print(run_game())