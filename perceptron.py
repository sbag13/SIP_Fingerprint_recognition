import tensorflow as tf
import sys, time

learning_rate = 0.5
tf_epochs = 20

inputSize = 64
L1Size = 64
outputSize = 51
std_dev = 0.03

x = tf.placeholder(tf.float64, [None, inputSize])
y = tf.placeholder(tf.float64, [None, outputSize])

W1 = tf.Variable(tf.random_normal([inputSize, L1Size], stddev=std_dev, dtype=tf.float64), name='W1')
b1 = tf.Variable(tf.random_normal([L1Size], dtype=tf.float64), name='b1')
W2 = tf.Variable(tf.random_normal([L1Size, outputSize], stddev=std_dev, dtype=tf.float64), name='W2')
b2 = tf.Variable(tf.random_normal([outputSize], dtype=tf.float64), name='b2')

L1 = tf.add(tf.matmul(x, W1), b1)
L1 = tf.nn.relu(L1)

y_ = tf.add(tf.matmul(L1, W2), b2)
y_ = tf.nn.softmax(y_)

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                        + (1 - y) * tf.log(1 - y_clipped), axis=1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

saver = tf.train.Saver()

def startLearning(inputMatrix, outputMatrix, network_name):
    with tf.Session() as sess:
        sess.run(init_op)
        learn(inputMatrix, outputMatrix, sess)
        print("accuracy: ", sess.run(accuracy, feed_dict={x: inputMatrix, y: outputMatrix}))
        save_path = saver.save(sess, network_name)
        # save_path = saver.save(sess, "./model.ckpt")
        print("Model saved in path: %s" % save_path)
        predict(inputMatrix, outputMatrix, "Learning data.", network_name)

def continueLearning(inputMatrix, outputMatrix, network_name):
    with tf.Session() as sess:
        saver.restore(sess, network_name)  
        learn(inputMatrix, outputMatrix, sess)
        save_path = saver.save(sess, network_name)
        # save_path = saver.save(sess, "./model.ckpt")
        print("Model saved in path: %s" % save_path)
        predict(inputMatrix, outputMatrix, "Learning data.", network_name)

def learn(inputMatrix, outputMatrix, sess):
    total_samples = int(len(inputMatrix))  
    for epoch in range(tf_epochs):
        start = time.time()
        avg_cost = 0
        for i in range(total_samples):
            sys.stdout.write("\r%d / %d   " % (i , total_samples))
            sys.stdout.flush()
            _, c = sess.run([optimiser, cross_entropy], 
                        feed_dict={x: inputMatrix, y: outputMatrix})
            avg_cost += c
        stop = time.time()
        avg_cost /= total_samples
        print("\nEpoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "   time: ", (stop - start))

def predict(inputMatrix, outputMatrix, data_name, model_path):
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        print(data_name)
        print("accuracy: ", sess.run(accuracy, feed_dict={x: inputMatrix, y: outputMatrix}))