import tensorflow as tf

class ConvNetLayer(object):
    def __init__(self, N, ntypes, imWidth, imWidthReduc):
        self.N = N
        self.ntypes = ntypes
        self.imWidth = imWidth
        self.imWidthReduc = imWidthReduc

    def build_layer(self, prevHPool, prevNumFeatures, numFeatures):
        W_conv = self.weight_variable([5, 5, prevNumFeatures, numFeatures])
        b_conv = self.bias_variable([numFeatures])
        h_conv = tf.nn.relu(self.conv2d(prevHPool, W_conv) + b_conv)
        h_pool = self.max_pool_2x2(h_conv)
        print h_pool

        return h_pool

    def connect_layers(self, h_pool, numFeatures, layerNum):
        W_fc = self.weight_variable([self.imWidthReduc/layerNum * self.imWidthReduc/layerNum * numFeatures, 1024])
        b_fc = self.bias_variable([1024])
        h_pool_flat = tf.reshape(h_pool, [-1, self.imWidthReduc/layerNum * self.imWidthReduc/layerNum * numFeatures])
        h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)

        return h_fc

    def dropout(self, h_fc):
        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(h_fc, keep_prob)

        return keep_prob, h_fc_drop

    def readout_layer(self):
        W_fc = self.weight_variable([1024, self.ntypes])
        b_fc = self.bias_variable([self.ntypes])

        return W_fc, b_fc


    # WEIGHT INITIALISATION
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # CONVOLUTION AND POOLING
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
