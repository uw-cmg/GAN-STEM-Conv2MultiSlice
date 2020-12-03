from __future__ import print_function, division
import scipy

#from keras.datasets import mnist
#from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
from data_loader_gray import DataLoader
import numpy as np
import tensorflow as tf
import os
plt.switch_backend('agg')

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'stem_gray_5'

        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()
        # self.generator = self.build_generator_flat()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[50, 50],
                              optimizer=optimizer)

        self.errors = []

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=7, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            print("d: ", d.shape.dims)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            print("u: ", u.shape.dims,"skip_input: ", skip_input.shape.dims)
            u = Concatenate()([u, skip_input])
            return u

        def conv2d_flat(layer_input, filters, f_size=7, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            print("d: ", d.shape.dims)
            return d

        def deconv2d_flat(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=1)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            print("u: ", u.shape.dims,"skip_input: ", skip_input.shape.dims)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        # u0 = deconv2d(d8, d7, self.gf * 8, f_size=1)
        # u1 = deconv2d(u0, d6, self.gf * 8)
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        # u2 = deconv2d(d6, d5, self.gf * 8)
        u3 = deconv2d(u2, d4, self.gf*8)
        # u3 = deconv2d(d5, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=7, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_generator_flat(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=7, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            print("d: ", d.shape.dims)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            print("u: ", u.shape.dims, "skip_input: ", skip_input.shape.dims)
            u = Concatenate()([u, skip_input])
            return u

        def conv2d_flat(layer_input, filters, f_size=7, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            print("d: ", d.shape.dims)
            return d

        def deconv2d_flat(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=1)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            print("u: ", u.shape.dims, "skip_input: ", skip_input.shape.dims)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d_flat(d0, self.gf, bn=False)
        d2 = conv2d_flat(d1, self.gf * 2)
        d3 = conv2d_flat(d2, self.gf * 4)
        d4 = conv2d_flat(d3, self.gf * 8)
        d5 = conv2d_flat(d4, self.gf * 8)
        d6 = conv2d_flat(d5, self.gf * 8)
        d7 = conv2d_flat(d6, self.gf * 8)
        d8 = conv2d_flat(d6, self.gf * 8)
        d9 = conv2d_flat(d6, self.gf * 8)
        d10 = conv2d_flat(d6, self.gf * 8)
        d11 = conv2d_flat(d6, self.gf * 8)

        # Upsampling
        # u0 = deconv2d(d8, d7, self.gf * 8, f_size=1)
        # u1 = deconv2d(u0, d6, self.gf * 8)
        u8 = deconv2d_flat(d11, d10, self.gf * 8)
        u9 = deconv2d_flat(u8, d9, self.gf * 8)
        u10 = deconv2d_flat(u9, d8, self.gf * 8)
        u11 = deconv2d_flat(u10, d7, self.gf * 8)
        u1 = deconv2d_flat(u11, d6, self.gf * 8)
        u2 = deconv2d_flat(u1, d5, self.gf * 8)
        # u2 = deconv2d(d6, d5, self.gf * 8)
        u3 = deconv2d_flat(u2, d4, self.gf * 8)
        # u3 = deconv2d(d5, d4, self.gf * 8)
        u4 = deconv2d_flat(u3, d3, self.gf * 4)
        u5 = deconv2d_flat(u4, d2, self.gf * 2)
        u6 = deconv2d_flat(u5, d1, self.gf)

        u7 = UpSampling2D(size=1)(u6)
        output_img = Conv2D(self.channels, kernel_size=7, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

        ''' plt.plot(range(len(self.errors)), self.errors, 'go')
        plt.title('Generator Error Over Time')
        plt.ylabel('RMSE')
        plt.xlabel('Epochs')
        plt.savefig("images/%s/%s.png" % (self.dataset_name, "rmse_graph"))
        plt.close()'''
        print('Error: ',min(self.errors))

        multi, imgs = self.data_loader.load_data(batch_size=35)
        fakes = self.generator.predict(imgs)
        residuals = multi - fakes

        for i in range(len(fakes)):
            np.save("images/102419/fakes/train/fake_%d.png" % (i), fakes[i])
            np.save("images/102419/multislice/train/multi_%d.png" % (i), multi[i])
            np.save("images/102419/convolution/train/conv_%d.png" % (i), imgs[i])
            np.save("images/102419/residuals/train/residual_%d.png" % (i), residuals[i])

        test_multi, test_imgs = self.data_loader.load_data(batch_size=11, is_testing=True)
        test_fakes = self.generator.predict(imgs)
        test_residuals = test_multi - test_fakes

        for i in range(len(fakes)):
            np.save("images/102419/fakes/test/fake_%d.png" % (i), test_fakes[i])
            np.save("images/102419/multislice/test/multi_%d.png" % (i), test_multi[i])
            np.save("images/102419/convolution/test/conv_%d.png" % (i), test_imgs[i])
            np.save("images/102419/residuals/test/residual_%d.png" % (i), test_residuals[i])

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3
        errors = []
        imgs_A, imgs_B = self.data_loader.load_data(batch_size=11, is_testing=True)
        fake_A = self.generator.predict(imgs_B)
        imgs_A = 0.5 * imgs_A + 0.5
        fake_A = 0.5 * fake_A + 0.5
        std = np.std(np.array(imgs_A))
        mean = np.mean(np.array(imgs_A))
        for i in range(11):
            rmse = self.calculate_rmse(fake_A[i], imgs_A[i])
            fs_rmse = rmse / std
            errors.append(fs_rmse)
        print('epoch fsrmse =', np.average(errors))

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)
        # gen_imgs stores a batch of convolutions, then a batch of generations, then a batch of multislice
        # e.g. if batch_size=3, the generated images will be at indices 3, 4, and 5
        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Convolution', '"Fake" Multislice', 'Multislice']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        # errors = []
        # std = np.std(np.array(gen_imgs[6:]))
        # mean = np.mean(np.array(gen_imgs[6:]))

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt][:, :, 0])
                axs[i, j].set_title(titles[i])
                if cnt in [3,4,5]:
                    fm_rmse = self.calculate_fm_rmse(gen_imgs[cnt], gen_imgs[cnt + 3])
                    fs_rmse = self.calculate_fs_rmse(gen_imgs[cnt], gen_imgs[cnt + 3])
                    rmse = self.calculate_rmse(gen_imgs[cnt], gen_imgs[cnt + 3])
                    # ssim_rating = compare_ssim(gen_imgs[cnt+3], gen_imgs[cnt], range=gen_imgs[cnt].max() - gen_imgs[cnt].min(), multichannel=True)
                    # errors.append(fs_rmse)
                    fig.text(0.5 + 0.29 * (j - 1), 0.07, "RMSE = " + str(rmse * 100)[:5], ha='center')
                    fig.text(0.5 + 0.29 * (j - 1), 0.04, "fmRMSE = " + str(rmse * 100 / mean)[:5] + "%", ha='center')
                    fig.text(0.5 + 0.29 * (j - 1), 0.01, "fsRMSE = " + str(rmse * 100 / std)[:5] + "%", ha='center')
                    '''axs[0, j].imshow(
                        (np.abs(gen_imgs[cnt] - gen_imgs[cnt + 3])) / (gen_imgs[cnt] - gen_imgs[cnt + 3]).max())'''
                axs[i,j].axis('off')
                cnt += 1
        self.errors.append(np.average(errors))
        # saves the whole plot
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        # save a full-resolution image from gen_imgs
        # scipy.misc.imsave("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i), gen_imgs[3])
        plt.close()

    def calculate_rmse(self, predictions, targets):
        row = 0
        column = 0
        while True:
            if targets[row] is not [0]*self.img_cols:
                break
            else:
                row += 1
        while True:
            if targets.T[column] is not [0]*self.img_rows:
                break
            else:
                column += 1

        return np.sqrt(np.mean((predictions[row:256-row, column:256-column] - targets[row:256-row, column:256-column]) ** 2))

    def calculate_gray_rmse(self, predictions, targets):
        grayPredictions = (predictions[:, :, 0] + predictions[:, :, 1] + predictions[:, :, 2])/3
        grayTargets = (targets[:, :, 0]*.299 + targets[:, :, 1]*.587 + targets[:, :, 2]*.114)
        return self.calculate_rmse(grayPredictions, grayTargets)

    def calculate_fm_rmse(self, predictions, targets):
        row = 0
        column = 0
        while True:
            if targets[row] is not [0] * self.img_cols:
                break
            else:
                row += 1
        while True:
            if targets.T[column] is not [0] * self.img_rows:
                break
            else:
                column += 1

        return np.sqrt(np.mean(
            (predictions[row:256 - row, column:256 - column] - targets[row:256 - row, column:256 - column]) ** 2)) / (
                   predictions[row:256 - row, column:256 - column].max()-predictions[row:256 - row, column:256 - column].min())

    def calculate_fs_rmse(self, predictions, targets):
        row = 0
        column = 0
        while True:
            if targets[row] is not [0] * self.img_cols:
                break
            else:
                row += 1
        while True:
            if targets.T[column] is not [0] * self.img_rows:
                break
            else:
                column += 1

        return np.sqrt(np.mean(
            (predictions[row:256 - row, column:256 - column] - targets[row:256 - row, column:256 - column]) ** 2)) / (
                predictions[row:256 - row, column:256 - column].std())

    def calculate_mae(self, predictions, targets):
        row = 0
        column = 0
        while True:
            if targets[row] is not [0] * self.img_cols:
                break
            else:
                row += 1
        while True:
            if targets.T[column] is not [0] * self.img_rows:
                break
            else:
                column += 1

        return np.mean(np.abs(predictions[row:256 - row, column:256 - column] - targets[row:256 - row, column:256 - column]))


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=10000, batch_size=8, sample_interval=100)
