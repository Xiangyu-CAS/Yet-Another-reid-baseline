import cv2
import random
import torch
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image


class GaussianBlur(object):
    """Apply gaussian blur with random parameters
    """

    def __init__(self, p, k):
        self.p = p
        assert k % 2 == 1
        self.k = k

    def __call__(self, input_tuple, *args, **kwargs):
        img, mask = input_tuple

        img = np.array(img)
        if float(torch.FloatTensor(1).uniform_()) < self.p:
            img = cv2.blur(img, (self.k, self.k))

        img = Image.fromarray(img)
        return img, mask


class RandomGrid(object):
    """Random grid
    """

    def __init__(self, p=0.15, color=-1, grid_size=(24, 64), thickness=(1, 1), angle=(0, 180), **kwargs):
        self.p = p
        self.color = color
        self.grid_size = grid_size
        self.thickness = thickness
        self.angle = angle

    def __call__(self, img, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return img

        if self.color == (-1, -1, -1):  # Random color
            color = tuple([random.randint(0, 256) for _ in range(3)])
        else:
            color = self.color

        grid_size = random.randint(*self.grid_size)
        thickness = random.randint(*self.thickness)
        angle = random.randint(*self.angle)

        return self.draw_grid(img, grid_size, color, thickness, angle)

    @staticmethod
    def draw_grid(image, grid_size, color, thickness, angle):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        mask = np.zeros((h * 8, w * 8, 3), dtype='uint8')
        mask_h, mask_w = mask.shape[:2]
        for i in range(0, mask_h, grid_size):
            p1 = (0, i)
            p2 = (mask_w, i + grid_size)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)
        for i in range(0, mask_w, grid_size):
            p1 = (i, 0)
            p2 = (i + grid_size, mask_h)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)

        center = (mask_w // 2, mask_h // 2)

        if angle > 0:
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            mask = cv2.warpAffine(mask, rot_mat, (mask_w, mask_h), flags=cv2.INTER_LINEAR)

        offset = (random.randint(-16, 16), random.randint(16, 16))
        center = (center[0] + offset[0], center[1] + offset[1])
        mask = mask[center[1] - h // 2: center[1] + h // 2, center[0] - w // 2: center[0] + w // 2, :]
        mask = cv2.resize(mask, (w, h))
        assert img.shape == mask.shape
        img = np.where(mask == 0, img, color).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)


class RandomFigures(object):
    """Insert random figure or some figures from the list [line, rectangle, circle]
    with random color and thickness
    """

    def __init__(self, p=0.33, random_color=True, always_single_figure=False,
                 thicknesses=(1, 6), circle_radiuses=(5, 64), figure_prob=0.5, **kwargs):
        self.p = p
        self.random_color = random_color
        self.always_single_figure = always_single_figure
        self.figures = (cv2.line, cv2.rectangle, cv2.circle)
        self.thicknesses = thicknesses
        self.circle_radiuses = circle_radiuses
        self.figure_prob = figure_prob

    def __call__(self, image):
        if random.uniform(0, 1) > self.p:
            return image

        if self.always_single_figure:
            figure = [self.figures[random.randint(0, len(self.figures) - 1)]]
        else:
            figure = []
            for i in range(len(self.figures)):
                if random.uniform(0, 1) > self.figure_prob:
                    figure.append(self.figures[i])

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = cv_image.shape[:2]
        for f in figure:
            p1 = (random.randint(0, w), random.randint(0, h))
            p2 = (random.randint(0, w), random.randint(0, h))
            color = tuple([random.randint(0, 256) for _ in range(3)]) if self.random_color else (0, 0, 0)
            thickness = random.randint(*self.thicknesses)
            if f != cv2.circle:
                cv_image = f(cv_image, p1, p2, color, thickness)
            else:
                r = random.randint(*self.circle_radiuses)
                cv_image = f(cv_image, p1, r, color, thickness)

        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img)


class RandomRotate(object):
    """Random rotate
    """

    def __init__(self, p=0.33, angle=(-5, 5), **kwargs):
        self.p = p
        self.angle = angle

    def __call__(self, img, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return img

        rnd_angle = random.randint(self.angle[0], self.angle[1])
        img = F.rotate(img, rnd_angle, resample=False, expand=False, center=None)

        return img