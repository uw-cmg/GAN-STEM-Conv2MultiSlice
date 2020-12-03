import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size, replace=False)

        imgs_A = []
        imgs_B = []
        max_vals = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]
            max_vals.append(img_A.max())

            # If training => do random flip
            ''' if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)'''

            # imgs_A.append(img_A)
            # imgs_B.append(img_B)
            imgs_A.append(img_A * 2. / img_A.max() - 1.)
            imgs_B.append(img_B * 2. / img_B.max() - 1.)

        ''' max_val = max(max_vals)
        imgs_A = np.array(imgs_A) / (max_val/2.) - 1.
        imgs_B = np.array(imgs_B) / (max_val/2.) - 1.
        print(imgs_B[0].max(), ",", imgs_B[0].min())
        print(imgs_A[0].max(), ",", imgs_A[0].min())'''
        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            max_vals = []
            for img in batch:
                img = self.imread(img)
                h, w, _ = img.shape
                half_w = int(w/2)
                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]
                max_vals.append(img_B.max())

                ''' if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)'''

                imgs_A.append(img_A*2. / img_A.max() - 1.)
                imgs_B.append(img_B*2. / img_B.max() - 1.)

            # max_val = max(max_vals)
            # imgs_A = np.array(imgs_A) / (max_val/2.) - 1.
            # imgs_B = np.array(imgs_B) / (max_val/2.) - 1.
            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)

            yield imgs_A, imgs_B

    def load_all_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        imgs_A = []
        imgs_B = []
        grids = []
        max_vals = []
        for i in range(len(path)):
            img = self.imread(path[i])
            grid = np.array(np.load('./datasets/%s/%s/*' % (self.dataset_name, data_type)))

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]
            max_vals.append(img_A.max())

            imgs_A.append(img_A * 2. / img_A.max() - 1.)
            imgs_B.append(img_B * 2. / img_B.max() - 1.)
            girds.append(grid)

        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)
        grids = np.array(grids)

        return imgs_A, imgs_B, grids

    def imread(self, path):
        # img = scipy.misc.imread(path).astype(np.float)
        img = np.array(np.load(path))
        newImg = np.ndarray(shape=(img.shape[0], img.shape[1], 1))
        newImg[:, :, 0] = img
        return newImg
