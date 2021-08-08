import tensorflow as tf
import math


def int_shape(x):
    return list(map(int, x.get_shape()))


class GatedConv2d(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size=(3, 3), stride=(1, 1), rate=1, pad='SAME', name='GatedConv2d',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.filter_size = filter_size
        self.conv = tf.keras.layers.Conv2D(num_filters, filter_size, strides=stride, padding=pad, dilation_rate=rate)
        self.elu = tf.keras.layers.ELU()

    def call(self, inputs, *args, **kwargs):
        output = self.conv(inputs)
        x, y = tf.split(output, 2, 3)
        x = self.elu(x)
        y = tf.nn.sigmoid(y)
        x = x * y
        return x


class GatedDeconv2d(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size=(3, 3), stride=(1, 1), rate=1, pad='SAME', name='GatedConv2d',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.filter_size = filter_size
        self.conv = tf.keras.layers.Conv2DTranspose(num_filters, filter_size, strides=stride,
                                                    padding=pad, dilation_rate=rate)
        self.elu = tf.keras.layers.ELU()

    def call(self, inputs, *args, **kwargs):
        output = self.conv(inputs)
        x, y = tf.split(output, 2, 3)
        x = self.elu(x)
        y = tf.nn.sigmoid(y)
        x = x * y
        return x


def attention_transfer(f, b1, b2, ksize=3, stride=1, fuse_k=3, softmax_scale=50., fuse=False):
    # extract patches from background feature maps with rate (1st scale)
    bs1 = tf.shape(b1)
    int_bs1 = b1.get_shape().as_list()
    w_b1 = tf.compat.v1.extract_image_patches(b1, [1, 4, 4, 1], [1, 4, 4, 1], [1, 1, 1, 1], padding='SAME')
    w_b1 = tf.reshape(w_b1, [int_bs1[0], -1, 4, 4, int_bs1[3]])
    w_b1 = tf.transpose(w_b1, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # extract patches from background feature maps with rate (2nd scale)
    bs2 = tf.shape(b2)
    int_bs2 = b2.get_shape().as_list()
    w_b2 = tf.compat.v1.extract_image_patches(b2, [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
    w_b2 = tf.reshape(w_b2, [int_bs2[0], -1, 2, 2, int_bs2[3]])
    w_b2 = tf.transpose(w_b2, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # use structure feature maps as foreground for matching and use background feature maps for reconstruction.
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    w_f = tf.compat.v1.extract_image_patches(f, [1, ksize, ksize, 1], [1, stride, stride, 1], [1, 1, 1, 1], padding='SAME')
    w_f = tf.reshape(w_f, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w_f = tf.transpose(w_f, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw

    w_f_groups = tf.split(w_f, int_fs[0], axis=0)
    w_b1_groups = tf.split(w_b1, int_bs1[0], axis=0)
    w_b2_groups = tf.split(w_b2, int_bs2[0], axis=0)
    y1 = []
    y2 = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw1_wi, raw2_wi in zip(f_groups, w_f_groups, w_b1_groups, w_b2_groups):
        # conv for compare
        wi = wi[0]  # (k,k,c,hw)
        onesi = tf.ones_like(wi)
        xxi = tf.nn.conv2d(tf.square(xi), onesi, strides=[1, 1, 1, 1], padding="SAME")  # (1,h,w,hw)
        wwi = tf.reduce_sum(tf.square(wi), axis=[0, 1, 2], keepdims=True)  # (1,1,1,hw)
        xwi = tf.nn.conv2d(xi, wi, strides=[1, 1, 1, 1], padding="SAME")  # (1,h,w,hw)
        di = xxi + wwi - 2 * xwi
        di_mean, di_var = tf.nn.moments(di, 3, keepdims=True)
        di_std = di_var ** 0.5
        yi = -1 * tf.nn.tanh((di - di_mean) / di_std)

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1] * fs[2], fs[1] * fs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], fs[1], fs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1] * fs[2], fs[1] * fs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1, 1, 1, 1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], fs[2], fs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], fs[1] * fs[2]])

        # softmax to match
        yi = tf.nn.softmax(yi * scale, 3)

        wi_center1 = raw1_wi[0]
        wi_center2 = raw2_wi[0]
        y1.append(tf.nn.conv2d_transpose(yi, wi_center1, tf.concat([[1], bs1[1:]], axis=0), strides=[1, 4, 4, 1]))
        y2.append(tf.nn.conv2d_transpose(yi, wi_center2, tf.concat([[1], bs2[1:]], axis=0), strides=[1, 2, 2, 1]))

    y1 = tf.concat(y1, axis=0)
    y2 = tf.concat(y2, axis=0)

    return y1, y2


def feature_loss(f, idx, embed, softmax_scale=10.):
    fs = f.get_shape().as_list()
    embedding_dim = fs[-1]
    num_embeddings = embed.get_shape().as_list()[-1]
    flat_f = tf.reshape(f, [-1, embedding_dim])
    d = (tf.reduce_sum(flat_f ** 2, 1, keepdims=True)
         - 2 * tf.matmul(flat_f, embed)
         + tf.reduce_sum(embed ** 2, 0, keepdims=True))
    d_mean, d_var = tf.nn.moments(d, 1, keepdims=True)
    d_std = d_var ** 0.5
    d_score = -1 * tf.nn.tanh((d - d_mean) / d_std)
    d_score = tf.reshape(d_score, fs[:-1] + [-1])
    encoding = tf.one_hot(idx, num_embeddings)
    ce = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_scale * d_score, labels=encoding)
    loss = tf.reduce_mean(ce)
    return loss
