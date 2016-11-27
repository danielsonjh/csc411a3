import numpy as np
import matplotlib.pyplot as plt
import time
from random import shuffle
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
        x = data['x']
        y = data['y']

        self.n_data = x.shape[0]

        start_time = time.time()

        # Separate data evenly among classes
        n_labels = np.max(y)
        n_train_in_label = int(self.n_data / n_labels * train_ratio)
        train_indices = []
        valid_indices = []
        for label in range(1, n_labels+1):
            label_indices = np.squeeze(np.argwhere(y == label))
            train_indices.extend(label_indices[:n_train_in_label])
            valid_indices.extend(label_indices[n_train_in_label:])

            SHOW_DATA = False
            if SHOW_DATA:
                fig = plt.figure()
                n_to_show = 100
                grid_size = np.ceil(np.sqrt(n_to_show))
                for i in range(0, n_to_show):
                    fig.add_subplot(grid_size, grid_size, i + 1).imshow(x[label_indices[i]])
                plt.show()

        # Shuffle training data
        shuffle(train_indices)

        self.__train_x = x[train_indices]
        self.__train_y = y[train_indices]
        print(np.bincount(self.__train_y))
        self.__valid_x = x[valid_indices]
        self.__valid_y = y[valid_indices]
        print(np.bincount(self.__valid_y))

        self.__train_y = self.__one_hot_encode_labels(self.__train_y)
        self.__valid_y = self.__one_hot_encode_labels(self.__valid_y)

        self.n_train = self.__train_x.shape[0]
        self.n_valid = self.__valid_x.shape[0]

        end_time = time.time()

        print 'Finished preparing training and validation sets. Took {0}s'.format(end_time - start_time)

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
        n = raw_labels.shape[0]
        labels = np.zeros((n, n_labels), dtype=np.uint8)
        for i in range(n):
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
        batch_x, batch_y, new_batch_counter = self.__process_next_batch(self.__test_x, self.__test_x, self.n_test,
                                                                        self.__test_batch_counter, batch_size)
        self.__test_batch_counter = new_batch_counter
        return batch_x

    @staticmethod
    def __process_next_batch(x, y, n, batch_counter, batch_size):
        start = batch_counter
        end = start + batch_size
        batch_x = x[start:end]
        batch_y = y[start:end]
        new_batch_counter = end if end < n else 0

        return batch_x, batch_y, new_batch_counter

dl = DataLoader()

if __name__ == '__main__':
    dl.prepare_train_val_data(0.9)