from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from DataLoader import DataLoader
from InstanceNormalization import InstanceNormalization
import tensorflow as tf
from tensorflow.keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

model = load_model('colorization_model.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
model.summary()

batch_size = 2

data_loader = DataLoader('/media/bonilla/HDD_2TB_basura/databases/all_flowers/', (128, 128))
im, x, y = data_loader.load_test_batch(batch_size=batch_size)
Xs_colorized = model.predict(x)

q_ab = np.load("pts_in_hull.npy")
q_ab = np.array([q for i, q in enumerate(q_ab) if i % 8 == 0])
epsilon = 1e-8
res = []
T = 0.48
for idx, X_colorized_org in enumerate(Xs_colorized):
    img_rows, img_cols, nb_q = X_colorized_org.shape
    X_colorized = X_colorized_org.reshape((img_rows * img_cols, nb_q))

    X_colorized = np.exp(np.log(X_colorized + epsilon) / T)
    X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

    q_a = q_ab[:, 0].reshape((1, 40))
    q_b = q_ab[:, 1].reshape((1, 40))

    X_a = np.sum(X_colorized * q_a, 1).reshape((img_rows, img_cols))
    X_b = np.sum(X_colorized * q_b, 1).reshape((img_rows, img_cols))

    X_a = X_a + 128
    X_b = X_b + 128
    out_lab = np.zeros((img_rows, img_cols, 3), dtype=np.int32)
    out_lab[:, :, 0] = x[idx, :, :, 0] * 255
    out_lab[:, :, 1] = X_a
    out_lab[:, :, 2] = X_b
    out_lab = out_lab.astype(np.uint8)
    out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
    res.append(out_rgb)

fig, axes = plt.subplots(batch_size, 2)
for i in range(batch_size):
    axes[i % batch_size, 0].imshow(x[i % batch_size], cmap='gray')
    axes[i % batch_size, 0].axis('off')
    # axes[i % batch_size, 1].imshow(im[i % batch_size] / 255.)
    # axes[i % batch_size, 1].axis('off')
    r = (res[i % batch_size] - np.min(res[i % batch_size])) / (
                np.max(res[i % batch_size]) - np.min(res[i % batch_size]))
    axes[i % batch_size, 1].imshow(r)
    axes[i % batch_size, 1].axis('off')

plt.show()
plt.close()
