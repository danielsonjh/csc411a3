import csv
from basic import *

x, y, weights, biases, keep_prob, pred = conv_net()

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    print "Restoring model..."
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print "Model restored from file: %s" % model_path

    dl.prepare_test_data()

    with open('submission.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['Id', 'Prediction'])
        id = 1
        while id <= n_test:
            batch_x = dl.next_test_batch(batch_size)
            batch_pred = sess.run(pred, feed_dict={x: batch_x, keep_prob: 1.0})
            batch_pred = tf.argmax(batch_pred, 1).eval() + 1
            for i in range(0, batch_pred.shape[0]):
                csvwriter.writerow([id, batch_pred[i]])
                id += 1