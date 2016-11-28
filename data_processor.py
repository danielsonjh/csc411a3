import os
import tensorflow as tf
import numpy as np
import Image
import matplotlib.pyplot as plt
import time

train_data_filename = 'train.npz'
train_data_temp_filename = 'train_temp.npz'
orig_train_data_filename = 'orig_train.npz'
test_data_filename = 'test.npz'


class DataProcessor:

    def __init__(self):
        pass

    def load_augment_and_save_train_data(self, resume_from_file=False):
        """
            Apply these operations to images in classes with low count until they match the max count
            1. flip LR (2)
            2. crop and resize different spots (5)
            3. adjust brightness (2)
            4. adjust contrast (2)
            5. adjust saturation (4)
        """

        x = self.__load_data('train')
        y = self.__load_labels()

        if resume_from_file:
            print('Resuming from file...')
            data = np.load(train_data_temp_filename)
            x = data['x']
            y = data['y']

        label_counts = np.bincount(y)
        max_label_count = max(label_counts)

        x_in_label = [[]] * len(label_counts)
        for label in range(1, len(label_counts)):
            filter_indices = np.squeeze(np.argwhere(y == label))
            x_in_label[label] = x[filter_indices]

        print(label_counts)
        print(x.shape)
        print(y.shape)

        flip_lr = lambda x_i: tf.image.flip_left_right(x_i).eval()
        o_size = x.shape[1]
        c_size = 84
        crop_base = lambda x_i, top_left_y, top_left_x: tf.image.resize_images(
            tf.image.crop_to_bounding_box(x_i, top_left_y, top_left_x, c_size, c_size), [o_size, o_size],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).eval()
        crop_tl = lambda x_i: crop_base(x_i, 0, 0)
        crop_tr = lambda x_i: crop_base(x_i, 0, o_size - c_size)
        crop_bl = lambda x_i: crop_base(x_i, o_size - c_size, 0)
        crop_br = lambda x_i: crop_base(x_i, o_size - c_size, o_size - c_size)
        crop_mid = lambda x_i: crop_base(x_i, (o_size - c_size) / 2, (o_size - c_size) / 2)
        bright_1 = lambda x_i: tf.image.adjust_brightness(x_i, 0.2).eval()
        bright_2 = lambda x_i: tf.image.adjust_brightness(x_i, -0.2).eval()
        contr_1 = lambda x_i: tf.image.adjust_contrast(x_i, 1.5).eval()
        contr_2 = lambda x_i: tf.image.adjust_contrast(x_i, 0.5).eval()
        sat_1 = lambda x_i: tf.image.adjust_saturation(x_i, 2.0).eval()
        sat_2 = lambda x_i: tf.image.adjust_saturation(x_i, 1.5).eval()
        sat_3 = lambda x_i: tf.image.adjust_saturation(x_i, 0.5).eval()
        sat_4 = lambda x_i: tf.image.adjust_saturation(x_i, 0.25).eval()

        # Plot augmented data
        SHOW_AUGS = False
        if SHOW_AUGS:
            with tf.Session():
                image = x[0]
                fig = plt.figure()
                # Flip
                fig.add_subplot(5, 5, 1).imshow(flip_lr(image))
                # Crop
                fig.add_subplot(5, 5, 6).imshow(crop_tl(image))
                fig.add_subplot(5, 5, 7).imshow(crop_tr(image))
                fig.add_subplot(5, 5, 8).imshow(crop_bl(image))
                fig.add_subplot(5, 5, 9).imshow(crop_br(image))
                fig.add_subplot(5, 5, 10).imshow(crop_mid(image))
                # Brightness
                fig.add_subplot(5, 5, 11).imshow(bright_1(image))
                fig.add_subplot(5, 5, 12).imshow(bright_2(image))
                # Contrast
                fig.add_subplot(5, 5, 16).imshow(contr_1(image))
                fig.add_subplot(5, 5, 17).imshow(contr_2(image))
                # Saturation
                fig.add_subplot(5, 5, 21).imshow(sat_1(image))
                fig.add_subplot(5, 5, 22).imshow(sat_2(image))
                fig.add_subplot(5, 5, 23).imshow(sat_3(image))
                fig.add_subplot(5, 5, 24).imshow(sat_4(image))

                plt.show()

        image_ops = [
            [flip_lr],
            [crop_tl, crop_tr, crop_bl, crop_br, crop_mid],
            [bright_1, bright_2],
            [contr_1, contr_2],
            [sat_1, sat_2, sat_3, sat_4]
        ]

        with tf.Session():
            # In each class
            for label in range(1, len(label_counts)):
                # Apply each op type
                for op_type in range(0, len(image_ops)):
                    print 'Applying op type {0} to class {1}'.format(op_type + 1, label)
                    start_time = time.time()

                    n_ops_in_type = len(image_ops[op_type])
                    # To every image
                    for i in range(0, len(x_in_label[label])):
                        op_index = 0
                        # Until all ops in the op type are done OR there are enough images in the class
                        while op_index < n_ops_in_type and label_counts[label] < max_label_count:
                            new_image = image_ops[op_type][op_index](x_in_label[label][i])
                            x = np.concatenate((x, np.asarray([new_image])))
                            y = np.concatenate((y, np.asarray([label])))
                            x_in_label[label] = np.concatenate((x_in_label[label], np.asarray([new_image])))
                            label_counts[label] += 1
                            op_index += 1

                    end_time = time.time()
                    print 'Took {0}s'.format(end_time - start_time)

                print 'Done augmenting class {0}'.format(label)
                np.savez_compressed(train_data_temp_filename, x=x, y=y)
                print('Saved temp data.')

        print(np.bincount(y))
        print(x.shape)
        print(y.shape)

        SHOW_MIXED_AUGS = False
        if SHOW_MIXED_AUGS:
            fig = plt.figure()
            label = 8
            n_in_label = len(x_in_label[label - 1])
            grid_size = np.ceil(np.sqrt(n_in_label))
            for i in range(0, n_in_label):
                fig.add_subplot(grid_size, grid_size, i + 1).imshow(x_in_label[label - 1][i])
            plt.show()

        print('Saving data...')
        np.savez_compressed(train_data_filename, x=x, y=y)
        print('Done')

    def load_and_save_orig_train_data(self):
        x = self.__load_data('train')
        y = self.__load_labels()
        print('Saving data...')
        np.savez_compressed(orig_train_data_filename, x=x, y=y)
        print('Done')

    def load_and_save_test_data(self):
        test_x = self.__load_data('val')
        print('Saving data...')
        np.savez_compressed(test_data_filename, x=test_x)
        print('Done')

    @staticmethod
    def __load_data(data_dir):
        filename_list = map(lambda f: os.path.join(data_dir, f), os.listdir(data_dir))
        filename_list.sort()
        # filename_list = filename_list[:100]
        image_list = []
        for filename in filename_list:
            image_list.append(np.asarray(Image.open(filename)))
        data = np.asarray(image_list)

        return data

    @staticmethod
    def __load_labels():
        raw_labels = np.loadtxt(open('train.csv', 'rb'), delimiter=',', usecols=(1,), skiprows=1).astype(int)
        # return raw_labels[:100]
        return raw_labels

if __name__ == '__main__':

    dp = DataProcessor()

    # dp.load_augment_and_save_train_data()
    # dp.load_augment_and_save_train_data(resume_from_file=True)
    # dp.load_and_save_test_data()
    dp.load_and_save_orig_train_data()