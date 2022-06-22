import pygame
import numpy as np
import math
import subprocess
import threading
import sys

running = True
grid = np.full((100, 100), False)

def run():
    p = subprocess.Popen(('cargo', 'run'), stdout=subprocess.PIPE)

    while(running):
        line = p.stdout.readline().decode().strip()
        
        if line != '':
            coords = [int(coord) for coord in line.split(',')]
            grid[coords[0]][coords[1]] = not grid[coords[0]][coords[1]]

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    tile_size = (math.floor(screen.get_width() / grid.shape[0]), math.floor(screen.get_height() / grid.shape[1]))
    threading.Thread(target=run).start()

    while running:
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x][y]:
                    pygame.draw.rect(screen, pygame.Color(0, 0, 0), (x * tile_size[0], y * tile_size[1], tile_size[0], tile_size[1]))
                else:
                    pygame.draw.rect(screen, pygame.Color(255, 255, 255), (x * tile_size[0], y * tile_size[1], tile_size[0], tile_size[1]))

        pygame.display.flip()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False