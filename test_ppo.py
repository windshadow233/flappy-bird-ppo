from game.flappy_bird import FlappyBird
from ppo.models import ActionModel
import torch
import os
import pygame
import sys
import random
import argparse

rand = random.SystemRandom()
seed = rand.getrandbits(64)


parser = argparse.ArgumentParser(description='Play Flappy Bird with a trained DQN model.')
parser.add_argument('--display', '-d', action='store_true', help='Display the game window')
parser.add_argument('--checkpoint', '-ckpt', type=str, default='ppo/models/best.pth')
parser.add_argument('--seed', '-s', type=int, default=seed, help='Random seed for reproducibility')
args = parser.parse_args()

random.seed(args.seed)
print("Seed: ", args.seed)

image_size = 84

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_model = ActionModel().to(device)
action_model.load_state_dict(torch.load(args.checkpoint, map_location=device))


def play(display=False, rand=True):
    action_model.eval()
    data = []
    reward_sum = 0

    if not display:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    else:
        os.environ.pop("SDL_VIDEODRIVER", None)

    game = FlappyBird()

    state, reward, terminal = game.step(0)

    while 1:
        prob = action_model(state[None].to(device))[0].exp()
        if rand:
            action_dist = torch.distributions.Categorical(prob)
            action = action_dist.sample().item()
        else:
            action = prob.argmax().item()

        next_state, reward, terminal = game.step(action, last_state=state)
        data.append((state, action, reward, next_state, terminal))
        reward_sum += reward

        state = next_state
        print(f"Reward: {round(reward_sum, 1)}", end='\r')
        if terminal:
            break
    return data, reward_sum


print("Reward: ", round(play(display=args.display, rand=False)[-1], 1))
pygame.quit()
sys.exit()
