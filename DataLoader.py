import cv2
import glob
import matplotlib.pyplot as plt
import os
import numpy as np


class DataLoader:
    def __init__(self, data_path, target_size):
        self.data_path = data_path
        self.target_size = target_size
        self.all_images = np.array(glob.glob(os.path.join(data_path, '*')))
        self.num_images = len(self.all_images)
        print(f'Found dataset with {self.num_images} images.')

    def load_image(self, path, loading: str, rands):
        image = cv2.imread(path)
        image = cv2.resize(image, self.target_size)
        if rands[0] >= 1:
            image = cv2.flip(image, rands[1])
        if rands[2] == 1:
            image = cv2.rotate(image, rands[3])
        l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
        if loading.lower() == 'y':
            image = cv2.merge([a, b])
        elif loading.lower() == 'x':
            image = np.expand_dims(l, axis=-1)
        return (image.astype('float32') - 127.5) / 127.5

    def load_batch(self, batch_size):
        indexes = np.random.randint(low=0, high=self.num_images, size=(batch_size,)).astype(int)
        rands = [int(np.random.randint(0, 3)), int(np.random.randint(-1, 2)), int(np.random.randint(0, 3)),
                 np.random.choice([cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])]
        batch_y = np.array([self.load_image(p, loading='y', rands=rands) for p in self.all_images[indexes]], 'float32')
        batch_x = np.array([self.load_image(p, loading='x', rands=rands) for p in self.all_images[indexes]], 'float32')
        return batch_x, batch_y
