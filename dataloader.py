import os
import numpy as np
import Image

n_labels = 8
n_data = 7000
n_train = 6800


class DataLoader:

    valid_x = []
    valid_y = []

    __train_x = []
    __train_y = []
    __batch_counter = 0

    def __init__(self):
        pass

    def prepare_train_val_data(self):
        all_x = self.__load_data('train')
        all_y = self.__load_labels()
        # TODO: Shuffle x,y pairs together?
        self.__train_x = all_x[:n_train]
        self.__train_y = all_y[:n_train]
        self.valid_x = all_x[n_train:]
        self.valid_y = all_y[n_train:]
        print('train')
        print(self.__train_x.shape)
        print(self.__train_y.shape)
        print('valid')
        print(self.valid_x.shape)
        print(self.valid_y.shape)

    def next_batch(self, batch_size):
        if n_train % batch_size != 0:
            raise ValueError('batch size ' + batch_size + ' does not fit into ' + n_train)
        start = self.__batch_counter % n_train
        end = start + batch_size
        self.__batch_counter += batch_size
        batch_x = self.__train_x[start:end]
        batch_y = self.__train_y[start:end]
        return batch_x, batch_y

    @staticmethod
    def __load_data(data_dir):
        filename_list = map(lambda f: os.path.join(data_dir, f), os.listdir(data_dir))
        filename_list.sort()
        image_list = []
        for filename in filename_list:
            image_list.append(np.asarray(Image.open(filename)))
        data = np.asarray(image_list)

        print(data.shape)

        return data

    @staticmethod
    def __load_labels():
        raw_labels = np.loadtxt(open("train.csv", "rb"), delimiter=",", usecols=(1,), skiprows=1).astype(int)
        # Convert raw labels to one-hot encoding
        labels = np.zeros((n_data, n_labels))
        for i in range(n_data):
            labels[i, raw_labels[i] - 1] = 1

        print(labels.shape)

        return labels


dl = DataLoader()
