import pygame
import math
import time
import random
from player import Player
from alien import Alien

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.background = pygame.image.load('background.png')
        self.player = Player(self, [math.floor(screen.get_width() / 2), screen.get_height() - 64])
        self.bullets = []
        self.aliens = []
        self.keys = {}
        self.last_alien = time.time()
        self.alien_spawn_delay = random.randint(1, 3) / 2
    
    def is_key_down(self, key):
        return key in self.keys and self.keys[key]
    
    def draw(self):
        self.screen.blit(self.background, (0, 0))
        self.player.draw()

        for bullet in self.bullets:
            bullet.draw()
        
        for alien in self.aliens:
            alien.draw()
    
    def update(self):
        self.player.update()
        out_bullets = []

        for bullet in self.bullets:
            if bullet.position[1] < 0:
                out_bullets.append(bullet)
            else:
                bullet.update()
        
        for bullet in out_bullets:
            del self.bullets[self.bullets.index(bullet)]
        
        dead_aliens = []

        for alien in self.aliens:
            alien.update()

            if not alien.alive:
                dead_aliens.append(alien)
        
        for alien in dead_aliens:
            del self.aliens[self.aliens.index(alien)]
        
        if time.time() - self.last_alien >= self.alien_spawn_delay:
            self.alien_spawn_delay = random.randint(1, 3) / 2
            self.aliens.append(Alien(self, [32, 32]))
            self.last_alien = time.time()
    
    def run(self):
        running = True

        while running:
            self.draw()
            self.update()
            pygame.display.flip()

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    self.keys[e.key] = True
                elif e.type == pygame.KEYUP:
                    self.keys[e.key] = False
            
            time.sleep(1/60)

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    game = Game(screen)
    game.run()