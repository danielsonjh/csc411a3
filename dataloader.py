import numpy as np
from data_processor import train_data_filename, test_data_filename


class DataLoader:

    n_data = None
    n_train = None
    n_valid = None
    n_test = None

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

    def prepare_train_val_data(self, train_ratio):
        data = np.load(train_data_filename)
        all_x = data['x']
        all_y = data['y']
        self.n_data = all_x.shape[0]
        self.n_train = int(self.n_data * train_ratio)
        self.n_valid = self.n_data - self.n_train
        self.__one_hot_encode_labels(all_y)
        # TODO: Shuffle x,y pairs together?
        self.__train_y = all_y[:self.n_train]
        self.__train_x = all_x[:self.n_train]
        self.__valid_x = all_x[self.n_train:]
        self.__valid_y = all_y[self.n_train:]
        print('train')
        print(self.__train_x.shape)
        print(self.__train_y.shape)
        print('valid')
        print(self.__valid_x.shape)
        print(self.__valid_y.shape)

    def prepare_test_data(self):
        data = np.load(test_data_filename)
        self.__test_x = data['x']
        self.n_test = self.__test_x.shape[0]
        print('test')
        print(self.__test_x.shape)

    def __one_hot_encode_labels(self, raw_labels):
        n_labels = np.max(raw_labels)
        labels = np.zeros((self.n_data, n_labels), dtype=np.uint8)
        for i in range(self.n_data):
            labels[i, raw_labels[i] - 1] = 1

        return labels

    def next_train_batch(self, batch_size):
        batch_x, batch_y, new_batch_counter = self.__process_next_batch(self.__train_x, self.__train_y, self.n_train,
                                                                        self.__train_batch_counter, batch_size)
        self.__train_batch_counter = new_batch_counter
        return batch_x, batch_y

    def next_valid_batch(self, batch_size):
        batch_x, batch_y, new_batch_counter = self.__process_next_batch(self.__valid_x, self.__valid_y, self.n_valid,
                                                                        self.__valid_batch_counter, batch_size)
        self.__valid_batch_counter = new_batch_counter
        return batch_x, batch_y

    def next_test_batch(self, batch_size):
        batch_x, batch_y, new_batch_counter = self.__process_next_batch(self.__test_x, self.__test_x, self.n_train,
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

dl = DataLoader()
