import tensorflow as tf

class C3D_Network(object):
    num_classes = 45
    channels = 3
    num_frames_per_clip = 16

    def __init__(self):
        pass

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





