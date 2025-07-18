import pygame
import os
import sys
from game.flappy_bird import FlappyBird


def play_human():
    os.environ.pop("SDL_VIDEODRIVER", None)
    game = FlappyBird()
    _, reward, terminal = game.next_frame(0)
    reward_sum = reward

    game.count_down(3)

    while True:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                action = 1  # flap

        _, reward, terminal = game.next_frame(action)
        reward_sum += reward

        print(f"Reward: {round(reward_sum, 1)}", end='\r')
        if terminal:
            break

    print("Game Over. Reward:", round(reward_sum, 1))


if __name__ == "__main__":
    play_human()
    pygame.quit()
    sys.exit()
