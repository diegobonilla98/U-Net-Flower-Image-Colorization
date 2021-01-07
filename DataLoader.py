import cv2
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import sklearn.neighbors as nn

nb_neighbors = 5


def get_soft_encoding(image_ab, nn_finder, nb_q):
    h, w = image_ab.shape[:2]
    a = np.ravel(image_ab[:, :, 0])
    b = np.ravel(image_ab[:, :, 1])
    ab = np.vstack((a, b)).T
    dist_neighb, idx_neigh = nn_finder.kneighbors(ab)
    sigma_neighbor = 5
    wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    y = np.zeros((ab.shape[0], nb_q))
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    y[idx_pts, idx_neigh] = wts
    y = y.reshape((h, w, nb_q))
    return y


class DataLoader:
    def __init__(self, data_path, target_size):
        self.data_path = data_path
        self.target_size = target_size
        self.all_images = np.array(glob.glob(os.path.join(data_path, '*')))
        self.num_images = len(self.all_images)
        print(f'Found dataset with {self.num_images} images.')

        q_ab = np.load("pts_in_hull.npy")
        q_ab = np.array([q for i, q in enumerate(q_ab) if i % 8 == 0])
        self.nb_q = q_ab.shape[0]
        self.nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    def load_image(self, path, loading: str, rands):
        image = cv2.imread(path)
        image = cv2.resize(image, self.target_size)
        if rands is not None:
            if rands[0] >= 1:
                image = cv2.flip(image, rands[1])
            if rands[2] == 1:
                image = cv2.rotate(image, rands[3])
        if loading.lower() == 'test':
            return image[:, :, ::-1]
        if loading.lower() == 'y':
            l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
            image = cv2.merge([a, b])
            image = image.astype('int32') - 128
            return get_soft_encoding(image, self.nn_finder, self.nb_q)
        elif loading.lower() == 'x':
            image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), axis=-1)
            return image.astype('float32') / 255.

    def load_batch(self, batch_size):
        indexes = np.random.randint(low=0, high=self.num_images, size=(batch_size,)).astype(int)
        rands = [int(np.random.randint(0, 3)), int(np.random.randint(-1, 2)), int(np.random.randint(0, 3)),
                 np.random.choice([cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])]
        batch_y = np.array([self.load_image(p, loading='y', rands=rands) for p in self.all_images[indexes]], 'float32')
        batch_x = np.array([self.load_image(p, loading='x', rands=rands) for p in self.all_images[indexes]], 'float32')
        return batch_x, batch_y

    def load_test_batch(self, batch_size):
        indexes = np.random.randint(low=0, high=self.num_images, size=(batch_size,)).astype(int)
        images = np.array([self.load_image(p, loading='test', rands=None) for p in self.all_images[indexes]], 'float32')
        batch_y = np.array([self.load_image(p, loading='y', rands=None) for p in self.all_images[indexes]], 'float32')
        batch_x = np.array([self.load_image(p, loading='x', rands=None) for p in self.all_images[indexes]], 'float32')
        return images, batch_x, batch_y

# class DataLoader:
#     def __init__(self, data_path, target_size):
#         self.data_path = data_path
#         self.target_size = target_size
#         self.all_images = np.array(glob.glob(os.path.join(data_path, '*')))
#         self.num_images = len(self.all_images)
#         print(f'Found dataset with {self.num_images} images.')
#
#     def load_image(self, path, loading: str, rands):
#         image = cv2.imread(path)
#         image = cv2.resize(image, self.target_size)
#         if rands[0] >= 1:
#             image = cv2.flip(image, rands[1])
#         if rands[2] == 1:
#             image = cv2.rotate(image, rands[3])
#         l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
#         if loading.lower() == 'y':
#             image = cv2.merge([a, b])
#         elif loading.lower() == 'x':
#             image = np.expand_dims(l, axis=-1)
#         return (image.astype('float32') - 127.5) / 127.5
#
#     def load_batch(self, batch_size):
#         indexes = np.random.randint(low=0, high=self.num_images, size=(batch_size,)).astype(int)
#         rands = [int(np.random.randint(0, 3)), int(np.random.randint(-1, 2)), int(np.random.randint(0, 3)),
#                  np.random.choice([cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])]
#         batch_y = np.array([self.load_image(p, loading='y', rands=rands) for p in self.all_images[indexes]], 'float32')
#         batch_x = np.array([self.load_image(p, loading='x', rands=rands) for p in self.all_images[indexes]], 'float32')
#         return batch_x, batch_y
