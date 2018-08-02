import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

class_label = {"Archery": 0, "BabyCrawling": 1, "BasketballShooting": 2, "BenchPress": 3, "Biking": 4,
               "BlowingCandles": 5, "Bowling": 6, "BreastStroke": 7, "CleanAndJerk": 8, "CliffDiving": 9,
               "Drumming": 10, "FrisbeeCatch": 11, "GolfSwing": 12, "Haircut": 13, "HeadMassage": 14,
               "HighJump": 15, "HorseRiding": 16, "Hug": 17, "HulaHoop": 18, "JumpRope": 19,
               "Kayaking": 20, "Kiss": 21, "Laugh": 22, "LongJump": 23, "Marching": 24,
               "PlayingCello": 25, "PlayingGuitar": 26, "PlayingPiano": 27, "PlayingViolin": 28, "PoleVault": 29,
               "PullUps": 30, "PushUps": 31, "ShakingHands": 32, "SitUp": 33, "Skiing": 34,
               "Skijet": 35, "SkyDiving": 36, "SoccerJuggling": 37, "SoccerShooting": 38, "Somersault": 39,
               "TaiChi": 40, "ThrowDiscus": 41, "TrampolineJumping": 42, "WalkingWithDog": 43, "WashingHair": 44}

class C3D_Network(object):
    num_classes = len(class_label.keys())
    pretrain_model_path = './models/pretrain/sports1m_finetuning_ucf101.model'

    def __init__(self, x, batchsize, dropout_prob=1, trainable=False):
        """
        构造函数
        :param x:
        :param batchsize:
        :param dropout_prob:
        :param trainable:
        """
        # 获取训练的参数
        self._reader = pywrap_tensorflow.NewCheckpointReader(self.pretrain_model_path)
        self._x = x
        self._batch_size = batchsize
        self._dropout_prob = dropout_prob
        self._parameters_config(trainable)

    def contruct_graph(self):
        # Convolution Layer 1
        conv1 = self._conv3d('conv1', self._x, self._paramter('var_name/wc1'), self._paramter('var_name/bc1'))
        conv1 = tf.nn.relu(conv1, 'relu1')
        pool1 = self._max_pool('pool1', conv1, k=1)

        # Convolution Layer 2
        conv2 = self._conv3d('conv2', pool1, self._paramter('var_name/wc2'), self._paramter('var_name/bc2'))
        conv2 = tf.nn.relu(conv2, 'relu2')
        pool2 = self._max_pool('pool2', conv2, k=2)

        # Convolution Layer 3,4
        conv3 = self._conv3d('conv3a', pool2, self._paramter('var_name/wc3a'), self._paramter('var_name/bc3a'))
        conv3 = tf.nn.relu(conv3, 'relu3a')
        conv3 = self._conv3d('conv3b', conv3, self._paramter('var_name/wc3b'), self._paramter('var_name/bc3b'))
        conv3 = tf.nn.relu(conv3, 'relu3b')
        pool3 = self._max_pool('pool3', conv3, k=2)

        # Convolution Layer 5,6
        conv4 = self._conv3d('conv4a', pool3, self._paramter('var_name/wc4a'), self._paramter('var_name/bc4a'))
        conv4 = tf.nn.relu(conv4, 'relu4a')
        conv4 = self._conv3d('conv4b', conv4, self._paramter('var_name/wc4b'), self._paramter('var_name/bc4b'))
        conv4 = tf.nn.relu(conv4, 'relu4b')
        pool4 = self._max_pool('pool4', conv4, k=2)

        # Convolution Layer 7,8
        conv5 = self._conv3d('conv5a', pool4, self._paramter('var_name/wc5a'), self._paramter('var_name/bc5a'))
        conv5 = tf.nn.relu(conv5, 'relu5a')
        conv5 = self._conv3d('conv5b', conv5, self._paramter('var_name/wc5b'), self._paramter('var_name/bc5b'))
        conv5 = tf.nn.relu(conv5, 'relu5b')
        pool5 = self._max_pool('pool5', conv5, k=2)

        # Fully connected layer
        pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
        dense1 = tf.reshape(pool5, [self._batch_size, self._weights['wd1'].get_shape().as_list()[0]])  # Reshape conv3 output to fit dense layer input
        dense1 = tf.matmul(dense1, self._weights['wd1']) + self._biases['bd1']

        dense1 = tf.nn.relu(dense1, name='fc1')  # Relu activation
        dense1 = tf.nn.dropout(dense1, self._dropout_prob)

        dense2 = tf.nn.relu(tf.matmul(dense1, self._weights['wd2']) + self._biases['bd2'], name='fc2')  # Relu activation
        dense2 = tf.nn.dropout(dense2, self._dropout_prob)

        # Output: class prediction
        out = tf.matmul(dense2, self._weights['out']) + self._biases['out']

        return out

    def _parameters_config(self, trainable):
        """
        参数配置
        :param trainable: 如果是true，有惩罚参数，否则无惩罚参数
        :return:
        """
        if trainable:
            punish_lambda = 0.0005
        else:
            punish_lambda = None

        with tf.variable_scope('var_name'):
            self._weights = {
                'wd1': self._variable_with_weight_decay('wd1', [8192, 4096], 0.04, punish_lambda),
                'wd2': self._variable_with_weight_decay('wd2', [4096, 1024], 0.04, punish_lambda),
                'out': self._variable_with_weight_decay('wout', [1024, self.num_classes], 0.04, punish_lambda)
            }
            self._biases = {
                'bd1': self._variable_with_weight_decay('bd1', [4096], 0.04, None),
                'bd2': self._variable_with_weight_decay('bd2', [1024], 0.04, None),
                'out': self._variable_with_weight_decay('bout', [self.num_classes], 0.04, None),
            }

    def _conv3d(self, name, input, filter, bias):
        """
        3D卷积
        :param name:
        :param input: shape = [batch, depth, width, height, channels]
        :param filter: shape = [depth, width, height, in_channels, out_channels]
        :param bias:
        :return:
        """
        res_conv = tf.nn.conv3d(input, filter, [1, 1, 1, 1, 1], padding="SAME", name=name)
        res = tf.nn.bias_add(res_conv, bias)
        return res
    def _max_pool(self, name, input, k):
        """
        3D max_pool
        :param name:
        :param input: shape = [batch, depth, width, height, channels]
        :param k:
        :return:
        """
        # ksize shape = [batch, depth, w, h, channels]
        return tf.nn.max_pool3d(input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding="SAME", name=name)

    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, punish_lambda=None):
        var = self._variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
        if punish_lambda is not None:
            weight_decay = tf.nn.l2_loss(var) * punish_lambda
            tf.add_to_collection('losses', weight_decay)
        return var

    def _paramter(self, name):
        """
        根据name从pretrain模型中加载数据
        :param name:
        :return:
        """
        assert name in self._reader.get_variable_to_shape_map().keys(), '找不着'+name+'参数'
        return self._reader.get_tensor(name)

