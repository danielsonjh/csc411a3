import tensorflow as tf
from dataloader import *

learning_rate = 0.001
training_epochs = 30
batch_size = 200
dropout = 1.0  # Dropout, probability to keep units

batches_per_epoch = n_train / batch_size
display_step = 50
train_logs_path = '/tmp/tensorflow_logs/basic_train'
valid_logs_path = '/tmp/tensorflow_logs/basic_valid'
model_path = 'basic.ckpt'


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv_net():
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    y = tf.placeholder(tf.float32, [None, 8])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    weights = {
        'wc1': tf.Variable(tf.random_normal([15, 15, 3, 64])),
        'wc2': tf.Variable(tf.random_normal([10, 10, 64, 64])),
        # fully connected (input dim, output dim)
        'wd1': tf.Variable(tf.random_normal([16 * 16 * 64, 800])),
        'wd2': tf.Variable(tf.random_normal([800, 800])),
        # inputs, outputs (class prediction)
        'out': tf.Variable(tf.random_normal([800, 8]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([weights['wc1'].get_shape().as_list()[3]])),
        'bc2': tf.Variable(tf.random_normal([weights['wc2'].get_shape().as_list()[3]])),
        'bd1': tf.Variable(tf.random_normal([weights['wd1'].get_shape().as_list()[1]])),
        'bd2': tf.Variable(tf.random_normal([weights['wd2'].get_shape().as_list()[1]])),
        'out': tf.Variable(tf.random_normal([weights['out'].get_shape().as_list()[1]]))
    }

    # Reshape input picture
    x_in = tf.reshape(x, shape=[-1, 128, 128, 3])
    x_in = tf.nn.avg_pool(x_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    tf.image_summary('x_in', x_in)

    # Convolution Layer
    conv1 = conv2d(x_in, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return x, y, weights, biases, keep_prob, out


def main(_):

    dl.prepare_train_val_data()

    # Construct model
    with tf.name_scope('Model'):
        x, y, weights, biases, keep_prob, pred = conv_net()

    # Define loss and optimizer
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    with tf.name_scope('Accuracy'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    tf.scalar_summary("loss", cost)
    tf.scalar_summary("accuracy", accuracy)
    tf.image_summary('wc1', tf.transpose(weights['wc1'], [3, 0, 1, 2]), max_images=30)
    merged_summary_op = tf.merge_all_summaries()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        train_summary_writer = tf.train.SummaryWriter(train_logs_path, graph=tf.get_default_graph())
        valid_summary_writer = tf.train.SummaryWriter(valid_logs_path, graph=tf.get_default_graph())

        step = 1
        # Keep training until reach max iterations
        while step < batches_per_epoch * training_epochs:
            batch_x, batch_y = dl.next_train_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x,
                                           y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc, summary = sess.run([cost, accuracy, merged_summary_op],
                                              feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                train_summary_writer.add_summary(summary, step)

                print("Batch " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

                valid_batch = 0
                valid_acc = 0
                n_valid_batches = n_valid / batch_size
                while valid_batch < n_valid_batches:
                    valid_batch += 1
                    batch_x, batch_y = dl.next_valid_batch(batch_size)
                    valid_batch_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                    valid_acc += valid_batch_acc
                valid_acc /= n_valid_batches
                valid_summary = tf.Summary()
                valid_summary.value.add(tag="accuracy", simple_value=valid_acc)
                valid_summary_writer.add_summary(valid_summary, step)
                print("Validation Accuracy= " + "{:.5f}".format(valid_acc))

            step += 1

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        print("Model saved in file: " + save_path)

        print "Run the command line:\n" \
              "--> tensorboard --logdir=/tmp/tensorflow_logs " \
              "\nThen open http://0.0.0.0:6006/ into your web browser"


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='/tmp/data',
    #                   help='Directory for storing data')
    # FLAGS = parser.parse_args()
    tf.app.run()