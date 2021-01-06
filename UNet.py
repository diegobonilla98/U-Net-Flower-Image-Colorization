from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Input, LeakyReLU, add, Dropout, \
    BatchNormalization, Activation, Lambda, UpSampling2D
from Conv2DSN import ConvSN2D
from Conv2DSNTranspose import ConvSN2DTranspose
from InstanceNormalization import InstanceNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from DataLoader import DataLoader
import tensorflow as tf
import cv2

import matplotlib.pyplot as plt
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


class UNet:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.input_tensor = Input(shape=(128, 128, 1))
        self.gf = 64
        self.channels = 2

        self.auto_encoder = self.build_model()
        self.auto_encoder.summary()
        plot_model(self.auto_encoder, to_file='colorization_model.png')

    def build_model(self):
        def conv2d(layer_input, filters=16, strides=1, name=None, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name)(layer_input)
            d = InstanceNormalization(name=name + "_bn")(d)
            d = Activation('relu')(d)
            return d

        def residual(layer_input, filters=16, strides=1, name=None, f_size=3):
            d = conv2d(layer_input, filters=filters, strides=strides, name=name, f_size=f_size)
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name + "_2")(d)
            d = InstanceNormalization(name=name + "_bn2")(d)
            d = add([d, layer_input])
            return d

        def conv2d_transpose(layer_input, filters=16, strides=1, name=None, f_size=4):
            u = UpSampling2D(size=(2, 2), interpolation='bilinear')(layer_input)
            u = Conv2D(filters, strides=strides, name=name, kernel_size=f_size, padding='same')(u)
            u = InstanceNormalization(name=name + "_bn")(u)
            u = Activation('relu')(u)
            return u

        # Image input
        c0 = self.input_tensor
        c1 = conv2d(c0, filters=self.gf, strides=1, name="g_e1", f_size=7)
        c2 = conv2d(c1, filters=self.gf * 2, strides=2, name="g_e2", f_size=3)
        c3 = conv2d(c2, filters=self.gf * 4, strides=2, name="g_e3", f_size=3)

        r1 = residual(c3, filters=self.gf * 4, name='g_r1')
        r2 = residual(r1, self.gf * 4, name='g_r2')
        r3 = residual(r2, self.gf * 4, name='g_r3')
        r4 = residual(r3, self.gf * 4, name='g_r4')
        r5 = residual(r4, self.gf * 4, name='g_r5')
        r6 = residual(r5, self.gf * 4, name='g_r6')
        r7 = residual(r6, self.gf * 4, name='g_r7')
        r8 = residual(r7, self.gf * 4, name='g_r8')
        r9 = residual(r8, self.gf * 4, name='g_r9')

        d1 = conv2d_transpose(r9, filters=self.gf * 2, f_size=4, strides=1, name='g_d1_dc')
        d2 = conv2d_transpose(d1, filters=self.gf, f_size=4, strides=1, name='g_d2_dc')

        output_img = ConvSN2D(self.channels, kernel_size=7, strides=1, padding='same', activation='tanh')(d2)

        return Model(inputs=[c0], outputs=[output_img])

    def get_model(self):
        return self.auto_encoder

    def plot_images(self, epoch, batch_size=4):
        x, y = self.data_loader.load_batch(batch_size=batch_size)
        res = self.auto_encoder.predict(x)

        fig, axes = plt.subplots(batch_size, 3)
        for i in range(batch_size):
            axes[i % batch_size, 0].imshow((x[i % batch_size] + 1) / 2, cmap='gray')
            axes[i % batch_size, 0].axis('off')
            xx = np.uint8((cv2.merge([x[i % batch_size], y[i % batch_size, :, :, 0], y[i % batch_size, :, :, 1]]) + 1) * 127.5)
            yy = np.uint8((cv2.merge([x[i % batch_size], res[i % batch_size, :, :, 0], res[i % batch_size, :, :, 1]]) + 1) * 127.5)
            axes[i % batch_size, 1].imshow(cv2.cvtColor(xx, cv2.COLOR_LAB2RGB))
            # axes[i % batch_size, 1].imshow(y[i % batch_size])
            axes[i % batch_size, 1].axis('off')
            axes[i % batch_size, 2].imshow(cv2.cvtColor(yy, cv2.COLOR_LAB2RGB))
            axes[i % batch_size, 2].axis('off')
        plt.savefig(f'./results/epoch_{epoch}.jpg')

    def compile_and_fit(self, epochs, batch_size):
        def rec_loss(y_true, y_pred):
            mse = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return mse
        losses = []
        initial_lr = 0.0002
        optimizer = Adam(lr=initial_lr, beta_1=0.5)
        self.auto_encoder.compile(optimizer=optimizer, loss=rec_loss)
        for epoch in range(epochs):
            x, y = self.data_loader.load_batch(batch_size=batch_size)
            loss = self.auto_encoder.train_on_batch(x, y)
            print(f'Epoch: {epoch}/{epochs}\tloss: {loss}')
            optimizer.learning_rate.assign(initial_lr * (0.43 ** epoch))
            if epoch % 50 == 0:
                self.plot_images(epoch)
            # losses.append(loss)

    def save_model(self):
        self.auto_encoder.save('colorization_model.h5')


data_loader = DataLoader('/media/bonilla/HDD_2TB_basura/databases/all_flowers/', (128, 128))
asr = UNet(data_loader)
asr.compile_and_fit(10000, 32)
asr.save_model()
