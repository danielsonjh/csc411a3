import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import pickle


def create_graph():
    with gfile.FastGFile('classify_image_graph_def.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(x):
    nb_features = 2048

    features = np.empty((x.shape[0], nb_features))

    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(x):
            if (ind % 100 == 0):
                print('Processing %s...' % ind)

            predictions = sess.run(next_to_last_tensor,
                                   {'DecodeJpeg:0': image})
            features[ind, :] = np.squeeze(predictions)

    return features


create_graph()

train_data = np.load('../train.npz')
train_x = train_data['x']
train_y = train_data['y']
train_x = extract_features(train_x)
pickle.dump(train_x, open('features', 'wb'))
pickle.dump(train_y, open('labels', 'wb'))

train_data2 = np.load('../train2.npz')
train_x2 = train_data2['x']
train_y2 = train_data2['y']
train_x2 = extract_features(train_x2)
pickle.dump(train_x2, open('features2', 'wb'))
pickle.dump(train_y2, open('labels2', 'wb'))

train_data3 = np.load('../train3.npz')
train_x3 = train_data3['x']
train_y3 = train_data3['y']
train_x3 = extract_features(train_x3)
pickle.dump(train_x3, open('features3', 'wb'))
pickle.dump(train_y3, open('labels3', 'wb'))

test_data = np.load('../test.npz')
test_x = test_data['x']

test_x = extract_features(test_x)
pickle.dump(test_x, open('test_features', 'wb'))


