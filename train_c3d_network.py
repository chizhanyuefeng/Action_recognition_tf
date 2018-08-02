import logging
import tensorflow as tf
from dataset import Dataset
from c3d_network import C3D_Network, class_label

class Train_C3D_Network(object):

    depth = 16
    img_size = 112
    learning_rate = 0.0001
    model_save_path = './models/test_model/model.ckpt'

    def __init__(self, batch_size=20, train_step=5000, depth=16):
        self.batch_size = batch_size
        self.train_step = train_step
        self.depth = depth

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

        saver = tf.train.Saver()

        data = Dataset(self.batch_size, self.depth, self.img_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(1, self.train_step+1):
                train_x, train_y = data.get_next_batch()

                sess.run(train_op, feed_dict={x: train_x, label: train_y})
                if step%1 ==0:
                    res = sess.run([total_loss, accuracy], feed_dict={x: train_x, label: train_y})
                    print('step:%d, accuracy: %6f, total loss: %6f' % (step, res[1], res[0]))
                if step%100 == 0:
                    save_path = saver.save(sess, self.model_save_path)
                    print('model saved at', save_path)


if __name__=="__main__":
    train = Train_C3D_Network(batch_size=10)
    train.train()