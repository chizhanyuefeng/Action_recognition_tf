import tensorflow as tf
from c3d_network import C3D_Network, class_label

class Train_C3D_Network(object):
    batch_size = 20
    depth = 16
    img_size = 112
    learning_rate = 0.0001
    train_step = 5000

    def __init__(self):
        pass

    def train(self):
        x = tf.placeholder(tf.float32, shape=[self.batch_size,
                                              self.depth,
                                              self.img_size,
                                              self.img_size,
                                              3])

        label = tf.placeholder(tf.float32, shape=[self.batch_size, len(class_label.keys())])
        network = C3D_Network(x, self.batch_size, dropout_prob=0.5, trainable=True)
        net_predict = network.contruct_graph()

        # 计算loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=net_predict)
        loss = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', loss)

        with tf.name_scope('total_loss'):
            total_loss = tf.add_n(tf.get_collection('losses'))

        with tf.name_scope('optimizer'):
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(total_loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(net_predict, 1), tf.argmax(label, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(self.train_step):
                train_x, train_y = get_next(self.batch_size)

                sess.run(train_op, feed_dict={x: train_x, y: train_y})
                if step%100 ==0:
                    res = sess.run([total_loss, accuracy], feed_dict={x: train_x, y: train_y})
                    print('accuracy: %6f ,total loss: %6f'%(res[1],res[0]))