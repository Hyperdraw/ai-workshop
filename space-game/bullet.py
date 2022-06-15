import pygame

sprite = pygame.image.load('bullet.png')

class Bullet:
    def __init__(self, game, position):
        self.game = game
        self.position = position
        self.update_rect()
        
    def update_rect(self):
        self.rect = pygame.Rect(self.position[0] - 16, self.position[1] - 32, 32, 32)
    
    def draw(self):
        self.game.screen.blit(sprite, (self.position[0] - 16, self.position[1] - 32))

    def update(self):
        self.position[1] -= 20
        self.update_rect()