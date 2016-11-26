import os
import numpy as np
import Image

n_labels = 8
n_data = 7000
n_train = 6000
n_valid = n_data - n_train
n_test = 970


class DataLoader:

    __valid_x = []
    __valid_y = []
    __valid_batch_counter = 0

    __train_x = []
    __train_y = []
    __train_batch_counter = 0

    __test_x = []
    __test_batch_counter = 0

    def __init__(self):
        pass

    def prepare_train_val_data(self):
        all_x = self.__load_data('train')
        all_y = self.__load_labels()
        # TODO: Shuffle x,y pairs together?
        self.__train_x = all_x[:n_train]
        self.__train_y = all_y[:n_train]
        self.__valid_x = all_x[n_train:]
        self.__valid_y = all_y[n_train:]
        print('train')
        print(self.__train_x.shape)
        print(self.__train_y.shape)
        print('valid')
        print(self.__valid_x.shape)
        print(self.__valid_y.shape)

    def prepare_test_data(self):
        self.__test_x = self.__load_data('val')
        print('test')
        print(self.__test_x.shape)

    def next_train_batch(self, batch_size):
        batch_x, batch_y, new_batch_counter = self.__process_next_batch(self.__train_x, self.__train_y, n_train,
                                                                        self.__train_batch_counter, batch_size)
        self.__train_batch_counter = new_batch_counter
        return batch_x, batch_y

    def next_valid_batch(self, batch_size):
        batch_x, batch_y, new_batch_counter = self.__process_next_batch(self.__valid_x, self.__valid_y, n_valid,
                                                                        self.__valid_batch_counter, batch_size)
        self.__valid_batch_counter = new_batch_counter
        return batch_x, batch_y

    def next_test_batch(self, batch_size):
        batch_x, batch_y, new_batch_counter = self.__process_next_batch(self.__test_x, self.__test_x, n_train,
                                                                        self.__test_batch_counter, batch_size)
        self.__test_batch_counter = new_batch_counter
        return batch_x

    @staticmethod
    def __process_next_batch(x, y, n, batch_counter, batch_size):
        start = batch_counter % n
        end = start + batch_size
        new_batch_counter = batch_counter + batch_size
        batch_x = x[start:end]
        batch_y = y[start:end]

        return batch_x, batch_y, new_batch_counter

    @staticmethod
    def __load_data(data_dir):
        filename_list = map(lambda f: os.path.join(data_dir, f), os.listdir(data_dir))
        filename_list.sort()
        image_list = []
        for filename in filename_list:
            image_list.append(np.asarray(Image.open(filename)))
        data = np.asarray(image_list)

        return data

    @staticmethod
    def __load_labels():
        raw_labels = np.loadtxt(open("train.csv", "rb"), delimiter=",", usecols=(1,), skiprows=1).astype(int)
        # Convert raw labels to one-hot encoding
        labels = np.zeros((n_data, n_labels), dtype=np.uint8)
        for i in range(n_data):
            labels[i, raw_labels[i] - 1] = 1

        return labels


dl = DataLoader()
