import pygame
import time
from bullet import Bullet

class Player:
    def __init__(self, game, position):
        self.game = game
        self.position = position
        self.sprite = pygame.image.load('player.png')
        self.last_bullet = time.time()
    
    def draw(self):
        self.game.screen.blit(self.sprite, (self.position[0] - 32, self.position[1] - 32))
    
    def update(self):
        if self.game.is_key_down(pygame.K_a):
            self.position[0] -= 8
        
        if self.game.is_key_down(pygame.K_d):
            self.position[0] += 8

        if self.game.is_key_down(pygame.K_SPACE) and time.time() - self.last_bullet >= 1:
            self.game.bullets.append(Bullet(self.game, [self.position[0], self.position[1] - 32]))
            self.last_bullet = time.time()

        self.position[0] = min(max(self.position[0], 32), self.game.screen.get_width() - 32)