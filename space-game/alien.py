import pygame

sprite = pygame.image.load('enemy.png')

class Alien:
    def __init__(self, game, position):
        self.game = game
        self.position = position
        self.update_rect()
        self.alive = True
    
    def update_rect(self):
        self.rect = pygame.Rect(self.position[0] - 32, self.position[1] - 32, 64, 64)

    def draw(self):
        if self.alive:
            self.game.screen.blit(sprite, [self.position[0] - 32, self.position[1] - 32])

    def update(self):
        if self.alive:
            self.position[0] += 5

            if self.position[0] > self.game.screen.get_width() - 32:
                self.position[0] = 32
                self.position[1] += 64
            
            self.update_rect()

            for bullet in self.game.bullets:
                if self.rect.colliderect(bullet.rect):
                    self.alive = False
                    break