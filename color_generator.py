import random

class ColorGenerator:
    def __init__(self):
        self.threshold = 50

    def generate_color(self, existing_colors, avoid_color=(255, 0, 0)):
        while True:
            color = tuple(random.randint(0, 255) for _ in range(3))
            if color not in existing_colors and all(abs(color[i] - avoid_color[i]) > self.threshold for i in range(3)):
                return color


