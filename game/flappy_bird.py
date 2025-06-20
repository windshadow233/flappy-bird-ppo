"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import sys
from itertools import cycle
import torch
import random
from pygame import Rect, time, display
from pygame.event import pump
from pygame.image import load
from pygame.surfarray import array3d, pixels_alpha
from pygame.transform import rotate
import numpy as np
import os
import pygame
from utils import pre_processing
from game.game_config import *

randint = random.randint


class FlappyBird(object):

    fps = 30
    pipe_velocity_x = -4

    # parameters for bird
    min_velocity_y = -8
    max_velocity_y = 10
    downward_speed = 1
    upward_speed = -9

    bird_index_generator = cycle([0, 1, 2, 1])
    image_size = 84

    def __init__(self):
        pygame.init()
        self.fps_clock = time.Clock()
        self.screen_width = 288
        self.screen_height = 512
        self.render = os.getenv('SDL_VIDEODRIVER') != 'dummy'

        if self.render:
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height))
            pygame.display.set_caption('RL Flappy Bird')
        else:
            pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
            self.screen = pygame.Surface(
                (self.screen_width, self.screen_height))

        for event in pygame.event.get():
            pass

        self.base_image = load('game/assets/sprites/base.png').convert_alpha()
        self.background_image = load(
            'game/assets/sprites/background-black.png').convert()
        self.pipe_images = [
            rotate(
                load('game/assets/sprites/pipe-green.png').convert_alpha(),
                180),
            load('game/assets/sprites/pipe-green.png').convert_alpha()
        ]
        self.bird_images = [
            load('game/assets/sprites/redbird-upflap.png').convert_alpha(),
            load('game/assets/sprites/redbird-midflap.png').convert_alpha(),
            load('game/assets/sprites/redbird-downflap.png').convert_alpha()
        ]
        self.bird_hitmask = [
            pixels_alpha(image).astype(bool) for image in self.bird_images
        ]
        self.pipe_hitmask = [
            pixels_alpha(image).astype(bool) for image in self.pipe_images
        ]

        self.iter = self.bird_index = self.score = 0

        self.bird_width = self.bird_images[0].get_width()
        self.bird_height = self.bird_images[0].get_height()
        self.pipe_width = self.pipe_images[0].get_width()
        self.pipe_height = self.pipe_images[0].get_height()

        self.bird_x = int(self.screen_width / 5)
        self.bird_y = int((self.screen_height - self.bird_height) / 2)

        self.base_x = 0
        self.base_y = self.screen_height * 0.79
        self.base_shift = self.base_image.get_width(
        ) - self.background_image.get_width()

        pipes = [self.generate_pipe(), self.generate_pipe()]
        pipes[0]["x_upper"] = pipes[0]["x_lower"] = self.screen_width
        pipes[1]["x_upper"] = pipes[1]["x_lower"] = self.screen_width * 1.5
        self.pipes = pipes

        self.current_velocity_y = 0
        self.is_flapped = False

    def generate_pipe(self):
        x = self.screen_width + 10
        gap_y = randint(2, 9) * 10 + int(self.base_y / 5)
        gap_size = randint(*GAP_SIZE_RANGE)
        return {
            "x_upper": float(x),
            "y_upper": float(gap_y - self.pipe_height),
            "x_lower": float(x),
            "y_lower": float(gap_y + gap_size)
        }

    def is_collided(self):
        # Check if the bird touch ground
        if self.bird_height + self.bird_y + 1 >= self.base_y:
            return True
        bird_bbox = Rect(self.bird_x, self.bird_y, self.bird_width,
                         self.bird_height)
        pipe_boxes = []
        for pipe in self.pipes:
            pipe_boxes.append(
                Rect(pipe["x_upper"], pipe["y_upper"], self.pipe_width,
                     self.pipe_height))
            pipe_boxes.append(
                Rect(pipe["x_lower"], pipe["y_lower"], self.pipe_width,
                     self.pipe_height))
            # Check if the bird's bounding box overlaps to the bounding box of any pipe
            if bird_bbox.collidelist(pipe_boxes) == -1:
                return False
            for i in range(2):
                cropped_bbox = bird_bbox.clip(pipe_boxes[i])
                min_x1 = cropped_bbox.x - bird_bbox.x
                min_y1 = cropped_bbox.y - bird_bbox.y
                min_x2 = cropped_bbox.x - pipe_boxes[i].x
                min_y2 = cropped_bbox.y - pipe_boxes[i].y
                if np.any(self.bird_hitmask[
                        self.bird_index][min_x1:min_x1 + cropped_bbox.width,
                                         min_y1:min_y1 + cropped_bbox.height] *
                          self.pipe_hitmask[i][min_x2:min_x2 +
                                               cropped_bbox.width,
                                               min_y2:min_y2 +
                                               cropped_bbox.height]):
                    return True
        return False

    def step(self, action, last_state=None):
        image, reward, terminal = self.next_frame(action)
        image = pre_processing(image[:self.screen_width, :int(self.base_y)],
                               self.image_size, self.image_size)
        image = torch.from_numpy(image)
        if last_state is not None:
            state = torch.cat((last_state[1:], image), dim=0)
        else:
            state = torch.tile(image, (4, 1, 1))
        return state, reward, terminal

    def count_down(self, seconds=3):
        font_size = 72
        font = pygame.font.SysFont(None, font_size)
        start_time = pygame.time.get_ticks()

        while True:
            elapsed = pygame.time.get_ticks() - start_time
            remaining = max(0, seconds - elapsed // 1000)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.render_screen()

            if remaining > 0:
                text = font.render(str(remaining), True, (255, 255, 255))
                rect = text.get_rect(center=(self.screen.get_width() // 2,
                                             self.screen.get_height() // 2))
                self.screen.blit(text, rect)
            else:
                break

            display.update()
            self.fps_clock.tick(self.fps)

    def render_screen(self):
        self.screen.blit(self.background_image, (0, 0))
        self.screen.blit(self.base_image, (self.base_x, self.base_y))
        self.screen.blit(self.bird_images[self.bird_index],
                         (self.bird_x, self.bird_y))
        for pipe in self.pipes:
            self.screen.blit(self.pipe_images[0],
                             (pipe["x_upper"], pipe["y_upper"]))
            self.screen.blit(self.pipe_images[1],
                             (pipe["x_lower"], pipe["y_lower"]))
        display.update()

    def next_frame(self, action):
        pump()
        reward = 0.1
        terminal = False
        # Check input action
        if action == 1:
            self.current_velocity_y = self.upward_speed
            self.is_flapped = True

        # Update score
        bird_center_x = self.bird_x + self.bird_width / 2
        for pipe in self.pipes:
            pipe_center_x = pipe["x_upper"] + self.pipe_width / 2
            if pipe_center_x < bird_center_x < pipe_center_x + 5:
                self.score += 1
                reward = 1
                break

        # Update index and iteration
        if (self.iter + 1) % 3 == 0:
            self.bird_index = next(self.bird_index_generator)
            self.iter = 0
        self.base_x = -((-self.base_x + 100) % self.base_shift)

        # Update bird's position
        if self.current_velocity_y < self.max_velocity_y and not self.is_flapped:
            self.current_velocity_y += self.downward_speed
        if self.is_flapped:
            self.is_flapped = False
        self.bird_y += min(
            self.current_velocity_y,
            self.bird_y - self.current_velocity_y - self.bird_height)
        if self.bird_y < 0:
            self.bird_y = 0

        # Update pipes' position
        for pipe in self.pipes:
            pipe["x_upper"] += self.pipe_velocity_x
            pipe["x_lower"] += self.pipe_velocity_x
        # Update pipes
        if 0 < self.pipes[0]["x_lower"] < 5:
            self.pipes.append(self.generate_pipe())
        if self.pipes[0]["x_lower"] < -self.pipe_width:
            del self.pipes[0]
        if self.is_collided():
            terminal = True
            reward = -1
            self.__init__()

        # Draw everything
        self.render_screen()
        image = array3d(self.screen)
        if self.render:
            self.fps_clock.tick(self.fps)
        return image, reward, terminal
