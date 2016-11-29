import csv
import tensorflow as tf
from data_loader import dl
from basic import conv_net, model_path, batch_size

x, y, weights, biases, keep_prob, pred = conv_net()

saver = tf.train.Saver()

with tf.Session() as sess:

    print "Restoring model..."
    saver.restore(sess, model_path)
    print "Model restored from file: %s" % model_path

    # Make predictions
    print('Making predictions...')
    dl.prepare_test_data()
    with open('submission.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['Id', 'Prediction'])
        id = 1
        while id <= dl.n_test:
            batch_x = dl.next_test_batch(batch_size)
            batch_pred = sess.run(pred, feed_dict={x: batch_x, keep_prob: 1.0})
            batch_pred = tf.argmax(batch_pred, 1).eval() + 1
            for i in range(0, batch_pred.shape[0]):
                csvwriter.writerow([id, batch_pred[i]])
                id += 1

    if False:
        print('Making train predictions...')
        dl.prepare_train_data_for_testing_prediction()
        with open('train_prediction_test.csv', 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['Id', 'Prediction', 'Usage'])
            id = 1
            while id <= dl.n_train:
                batch_x, batch_y = dl.next_train_batch(batch_size)
                batch_pred = sess.run(pred, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                batch_pred = tf.argmax(batch_pred, 1).eval() + 1
                for i in range(0, batch_pred.shape[0]):
                    csvwriter.writerow([id, batch_pred[i], 'Public'])
                    id += 1



