import os 
import pygame

class SokobanVisualizer:
    def __init__(self):
        self.window_size = 1024
        self.screen_width = 600
        self.screen_height = 600
        self.window = None
        self.screen = None
        self.clock = None
        self.isopen = True

    def render(self, obs):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self._load_sprites()

        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((255, 255, 255))
        self.pix_square_size = self.window_size / obs.shape[0]

        # Draw the game state on the canvas
        for x in range(obs.shape[0]):
            for y in range(obs.shape[1]):
                cell = obs[x, y]
                if cell[0] == 1:  # Wall
                    sprite = self.sprites['wall']
                elif cell[1] == 1:  # Floor
                    sprite = self.sprites['floor']
                elif cell[2] == 1:  # Box
                    sprite = self.sprites['box']
                elif cell[3] == 1:  # Box on target
                    sprite = self.sprites['box_on_target']
                elif cell[4] == 1:  # Player
                    sprite = self.sprites['player']
                elif cell[5] == 1:  # Player on target
                    sprite = self.sprites['player_on_target']
                elif cell[6] == 1:  # Target
                    sprite = self.sprites['target']
                sprite = pygame.transform.scale(sprite, (int(self.pix_square_size), int(self.pix_square_size)))
                self.canvas.blit(sprite, (y * self.pix_square_size, x * self.pix_square_size))

        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(2)

    def _load_sprites(self):
        img_path = os.path.join(os.path.dirname(__file__), 'surface')
        self.sprites = {
            'wall': pygame.image.load(os.path.join(img_path, 'wall.bmp')),
            'floor': pygame.image.load(os.path.join(img_path, 'floor.bmp')),
            'target': pygame.image.load(os.path.join(img_path, 'box_target.bmp')),
            'box': pygame.image.load(os.path.join(img_path, 'box.bmp')),
            'box_on_target': pygame.image.load(os.path.join(img_path, 'box_on_target.bmp')),
            'player': pygame.image.load(os.path.join(img_path, 'player.bmp')),
            'player_on_target': pygame.image.load(os.path.join(img_path, 'player_on_target.bmp')),
        }