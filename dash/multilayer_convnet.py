import tensorflow as tf

class ConvNetLayer(object):
    def __init__(self, N, ntypes, imWidth, imWidthReduc):
        self.N = N
        self.ntypes = ntypes
        self.imWidth = imWidth
        self.imWidthReduc = imWidthReduc

    def build_layer(self, prevHPool, prevNumFeatures, numFeatures):
        W_conv = self._weight_variable([5, 5, prevNumFeatures, numFeatures])
        b_conv = self._bias_variable([numFeatures])
        h_conv = tf.nn.relu(self._conv2d(prevHPool, W_conv) + b_conv)
        h_pool = self._max_pool_2x2(h_conv)
        # print(h_pool)

        return h_pool

    def connect_layers(self, h_pool, numFeatures, layerNum):
        W_fc = self._weight_variable([int(self.imWidthReduc/layerNum * self.imWidthReduc/layerNum * numFeatures), 1024])
        b_fc = self._bias_variable([1024])
        h_pool_flat = tf.reshape(h_pool, [-1, int(self.imWidthReduc/layerNum * self.imWidthReduc/layerNum * numFeatures)])
        h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)

        return h_fc

    def dropout(self, h_fc):
        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

        return keep_prob, h_fc_drop

    def readout_layer(self):
        W_fc = self._weight_variable([1024, self.ntypes])
        b_fc = self._bias_variable([self.ntypes])

        return W_fc, b_fc

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convnet_variables(imWidth, imWidthReduc, N, ntypes):
    x = tf.placeholder(tf.float32, shape=[None, N])
    y_ = tf.placeholder(tf.float32, shape=[None, ntypes])

    x_image = tf.reshape(x, [-1, imWidth, imWidth, 1])

    Layer1 = ConvNetLayer(N, ntypes, imWidth, imWidthReduc)
    h_pool1 = Layer1.build_layer(x_image, 1, 32)

    Layer2 = ConvNetLayer(N, ntypes, imWidth, imWidthReduc)
    h_pool2 = Layer2.build_layer(h_pool1, 32, 64)
    h_fc1 = Layer2.connect_layers(h_pool2, 64, 1)

    Layer3 = ConvNetLayer(N, ntypes, imWidth, imWidthReduc)
    h_pool3 = Layer3.build_layer(h_pool2, 64, 64)
    h_fc2 = Layer3.connect_layers(h_pool3, 64, 2)

    # READOUT LAYER
    keep_prob, h_fc2_drop = Layer3.dropout(h_fc1)
    W_fc3, b_fc3 = Layer3.readout_layer()
    y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

    return x, y_, keep_prob, y_conv
