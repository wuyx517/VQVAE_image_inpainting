from abc import ABC

import tensorflow as tf
import numpy as np


def int_shape(x):
    return list(map(int, x.get_shape()))


def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], 1, xs[2], xs[3]]), x[:, :xs[1] - 1, :, :]], 1)


def hw_flatten(x):
    return tf.reshape(x, [x.shape[0], -1, x.shape[-1]])


def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], xs[1], 1, xs[3]]), x[:, :, :xs[2] - 1, :]], 2)


class DownShiftedConv2d(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size=(2, 3), dilation_rate=(1, 1), kernel_regularizer='L2', padding='VALID',
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(filter_size, int):
            self.filter_size = (filter_size, filter_size)
        else:
            self.filter_size = filter_size
        self.wnconv = tf.keras.layers.Conv2D(num_filters, filter_size, strides=dilation_rate,
                                             kernel_regularizer=kernel_regularizer, padding=padding)

    def call(self, inputs, *args, **kwargs):
        output = tf.pad(inputs, [[0, 0], [self.filter_size[0] - 1, 0],
                                 [int((self.filter_size[1] - 1) / 2), int((self.filter_size[1] - 1) / 2)],
                                 [0, 0]])
        output = self.wnconv(output)
        return output


class DownRightShiftedConv2d(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size=(2, 2), dilation_rate=(1, 1), kernel_regularizer='L2', padding='VALID',
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(filter_size, int):
            self.filter_size = (filter_size, filter_size)
        else:
            self.filter_size = filter_size
        self.wnconv = tf.keras.layers.Conv2D(num_filters, filter_size, strides=dilation_rate,
                                             kernel_regularizer=kernel_regularizer, padding=padding)

    def call(self, inputs, *args, **kwargs):
        output = tf.pad(inputs, [[0, 0], [self.filter_size[0] - 1, 0], [self.filter_size[1] - 1, 0], [0, 0]])
        output = self.wnconv(output)
        return output


class NIN(tf.keras.layers.Layer):
    def __init__(self, num_units, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.wndense = tf.keras.layers.Dense(num_units, kernel_regularizer='L2')

    def call(self, inputs, *args, **kwargs):
        s = int_shape(inputs)
        x = tf.reshape(inputs, [np.prod(s[:-1]), s[-1]])
        x = self.wndense(x)
        return tf.reshape(x, s[:-1] + [self.num_units])


class OutResnet(tf.keras.layers.Layer):
    def __init__(self, num_filters, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.conv = NIN(num_filters)
        self.elu = tf.keras.layers.ELU()

    def call(self, inputs, *args, **kwargs):
        output = self.conv(self.elu(inputs))
        return inputs + output


class GatedResnet(tf.keras.layers.Layer):
    def __init__(self, a=None, h=None, num_res_filters=128, rate=1, dropout_p=0., causal_attention=False, num_head=8,
                 num_filters=32, conv='wnconv2d', **kwargs):
        super().__init__(**kwargs)
        self.causal_attention = causal_attention
        self.num_head = num_head
        if conv == 'wnconv2d':
            self.conv1 = tf.keras.layers.Conv2D(num_res_filters, 3, dilation_rate=1, padding='same',
                                                kernel_regularizer='L2')
            self.conv2 = tf.keras.layers.Conv2D(num_res_filters * 2, 3, dilation_rate=1, padding='same',
                                                kernel_regularizer='L2')
        elif conv == 'DownShiftedConv2d':
            self.conv1 = DownShiftedConv2d(num_res_filters, 3, dilation_rate=1)
            self.conv2 = DownShiftedConv2d(num_res_filters * 2, 3, dilation_rate=1)
        elif conv == 'DownRightShiftedConv2d':
            self.conv1 = DownRightShiftedConv2d(num_res_filters, 3, dilation_rate=1)
            self.conv2 = DownRightShiftedConv2d(num_res_filters * 2, 3, dilation_rate=1)
        else:
            self.conv1 = tf.keras.layers.Conv2D(num_res_filters, 3, dilation_rate=1)
            self.conv2 = tf.keras.layers.Conv2D(num_res_filters * 2, 3, dilation_rate=1)
        self.activity_1 = tf.keras.layers.ELU()
        self.nin_1 = NIN(num_res_filters)
        self.activity_2 = tf.keras.layers.ELU()

        self.nin_2 = NIN(num_filters * 8)
        self.activity_3 = tf.keras.layers.ELU()

        self.nin_q = NIN(num_filters // 8)
        self.nin_k = NIN(num_filters // 8)
        self.nin_v = NIN(num_filters // 2)
        self.nin_3 = NIN(num_filters * 4)
        self.dropout = tf.keras.layers.Dropout(dropout_p)

    def call(self, inputs, *args, **kwargs):
        if isinstance(inputs, list):
            if len(inputs) == 3:
                x = inputs[0]
                a = inputs[1]
                h = inputs[2]
            elif len(inputs) == 2:
                x = inputs[0]
                a = inputs[1]
                h = None
            else:
                print('Input shape is error')
                exit()
        else:
            x = inputs
            a = None
            h = None
        xs = int_shape(x)
        num_filters = xs[-1]

        c1 = self.conv1(self.activity_1(x))
        if a is not None:
            c1 += self.nin_1(self.activity_2(a))

        c2 = self.conv2(c1)
        if h is not None:
            c2 += self.nin_2(self.activity_3(h))

        a, b = tf.split(c2, 2, 3)
        c3 = a * tf.nn.sigmoid(b)

        if self.causal_attention:
            canvas_size = int(np.prod(int_shape(c3)[1:-1]))
            causal_mask = np.zeros([canvas_size, canvas_size], dtype=np.float32)
            for i in range(canvas_size):
                causal_mask[i, :i] = 1.
            causal_mask = tf.constant(causal_mask, dtype=tf.float32)
            causal_mask = tf.expand_dims(causal_mask, axis=0)

            multihead_src = []
            for head_rep in range(self.num_head):
                query = self.nin_q(c3)
                key = self.nin_k(c3)
                value = self.nin_v(c3)

                dot = tf.matmul(hw_flatten(query), hw_flatten(key), transpose_b=True) / np.sqrt(num_filters // 8)
                dot = dot - (1. - causal_mask) * 1e10  # masked softmax
                dot = dot - tf.reduce_max(dot, axis=-1, keepdims=True)
                causal_exp_dot = tf.exp(dot) * causal_mask
                causal_probs = causal_exp_dot / (tf.reduce_sum(causal_exp_dot, axis=-1, keepdims=True) + 1e-10)
                atten = tf.matmul(causal_probs, hw_flatten(value))
                atten = tf.reshape(atten, [xs[0], xs[1], xs[2], -1])
                multihead_src.append(atten)

            multihead = tf.concat(multihead_src, axis=-1)
            multihead = self.nin_3(multihead)
            c3 = c3 + multihead
            c3 = self.dropout(c3)

        return x + c3


class StructureCondition(tf.keras.layers.Layer):
    def __init__(self, mask, nr_channel=32, nr_res_channel=32, name='Structure Condition', **kwargs):
        super().__init__(name, **kwargs)
        self.mask = mask
        self.wnconv1 = tf.keras.layers.Conv2D(nr_channel, 5, kernel_regularizer='L2', padding='same')
        self.GatedResnet1 = GatedResnet(num_res_filters=nr_res_channel, num_filters=nr_channel)
        self.wnconv2 = tf.keras.layers.Conv2D(2 * nr_channel, 3, kernel_regularizer='L2', strides=(2, 2),
                                              padding='same')
        self.GatedResnet2 = GatedResnet(num_res_filters=2 * nr_res_channel, num_filters=2 * nr_channel)
        self.wnconv3 = tf.keras.layers.Conv2D(4 * nr_channel, 3, kernel_regularizer='L2', strides=(2, 2),
                                              padding='same')
        self.GatedResnet3 = GatedResnet(num_res_filters=4 * nr_res_channel, num_filters=4 * nr_channel)
        self.wnconv4 = tf.keras.layers.Conv2D(8 * nr_channel, 3, kernel_regularizer='L2', strides=(2, 2),
                                              padding='same')
        self.GatedResnet4 = GatedResnet(num_res_filters=8 * nr_res_channel, num_filters=8 * nr_channel)
        self.GatedResnet5 = GatedResnet(num_res_filters=8 * nr_res_channel, num_filters=8 * nr_channel)
        self.GatedResnet6 = GatedResnet(num_res_filters=8 * nr_res_channel, num_filters=8 * nr_channel)
        self.GatedResnet7 = GatedResnet(num_res_filters=8 * nr_res_channel, num_filters=8 * nr_channel, rate=2)
        self.GatedResnet8 = GatedResnet(num_res_filters=8 * nr_res_channel, num_filters=8 * nr_channel, rate=4)
        self.GatedResnet9 = GatedResnet(num_res_filters=8 * nr_res_channel, num_filters=8 * nr_channel, rate=8)
        self.GatedResnet10 = GatedResnet(num_res_filters=8 * nr_res_channel, num_filters=8 * nr_channel, rate=16)
        self.GatedResnet11 = GatedResnet(num_res_filters=8 * nr_res_channel, num_filters=8 * nr_channel)
        self.GatedResnet12 = GatedResnet(num_res_filters=8 * nr_res_channel, num_filters=8 * nr_channel)
        self.GatedResnet13 = GatedResnet(num_res_filters=8 * nr_res_channel, num_filters=8 * nr_channel)

    def call(self, inputs, *args, **kwargs):
        ones_x = tf.ones_like(inputs)[:, :, :, 0:1]
        inputs = tf.concat([inputs, ones_x, ones_x * self.mask], 3)

        output = self.wnconv1(inputs)
        output = self.GatedResnet1(output)
        output = self.wnconv2(output)
        output = self.GatedResnet2(output)
        output = self.wnconv3(output)
        output = self.GatedResnet3(output)
        output = self.wnconv4(output)
        output = self.GatedResnet4(output)
        output = self.GatedResnet5(output)
        output = self.GatedResnet6(output)
        output = self.GatedResnet7(output)
        output = self.GatedResnet8(output)
        output = self.GatedResnet9(output)
        output = self.GatedResnet10(output)
        output = self.GatedResnet11(output)
        output = self.GatedResnet12(output)
        output = self.GatedResnet13(output)
        return output


class StructurePixelcnn(tf.keras.layers.Layer):
    def __init__(self, h=None, ema=None, dropout_p=0., nr_resnet=20, nr_OutResnet=20, nr_channel=128,
                 nr_res_channel=128, nr_attention=4, nr_head=8, num_embeddings=512,
                 resnet_nonlinearity=tf.keras.layers.ELU,
                 name='StructurePixelcnn'):
        super().__init__(name)
        self.h = h
        self.down_shifted_conv2d_1 = DownShiftedConv2d(nr_channel)
        self.down_shifted_conv2d_2 = DownShiftedConv2d(nr_channel, filter_size=(1, 3))
        self.down_right_shifted_conv2d_1 = DownRightShiftedConv2d(nr_channel, filter_size=(2, 1))

        self.nr_attention_list = []
        for attn_rep in range(nr_attention):
            for rep in range(nr_resnet // nr_attention - 1):
                gated1 = GatedResnet(nr_channel, conv='DownShiftedConv2d', num_res_filters=nr_res_channel,
                                     dropout_p=dropout_p, num_head=nr_head)
                gated2 = GatedResnet(nr_channel, conv='DownRightShiftedConv2d', num_res_filters=nr_res_channel,
                                     dropout_p=dropout_p, num_head=nr_head)
                self.nr_attention_list.append(gated1)
                self.nr_attention_list.append(gated2)

            gated3 = GatedResnet(nr_channel, conv='DownShiftedConv2d', num_res_filters=nr_res_channel,
                                 dropout_p=dropout_p, num_head=nr_head)
            gated4 = GatedResnet(nr_channel, conv='DownRightShiftedConv2d', num_res_filters=nr_res_channel,
                                 dropout_p=dropout_p, num_head=nr_head,
                                 causal_attention=True)
            self.nr_attention_list.append(gated3)
            self.nr_attention_list.append(gated4)

        self.nr_OutResnet_list = []
        for _ in range(nr_OutResnet):
            ul = OutResnet(nr_channel)
            self.nr_OutResnet_list.append(ul)

        self.elu = tf.keras.layers.ELU()
        self.nin = NIN(num_embeddings)

    def call(self, inputs, *args, **kwargs):
        x = inputs[0]
        h = inputs[1]
        inputs_s = int_shape(x)
        e_pad = tf.concat([x, tf.ones(inputs_s[:-1] + [1])], 3)
        u = down_shift(self.down_shifted_conv2d_1(e_pad))
        ul = down_shift(self.down_shifted_conv2d_2(e_pad)) + right_shift(self.down_right_shifted_conv2d_1(e_pad))
        for i in range(len(self.nr_attention_list))[::2]:
            u = self.nr_attention_list[i]([u, None, h])
            ul = self.nr_attention_list[i + 1]([ul, u, h])

        for out_rep in self.nr_OutResnet_list:
            ul = out_rep(ul)

        e_out = self.nin(self.elu(ul))
        return e_out


class Structure(tf.keras.Model):
    def __init__(self, config, mask, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.structure_condition = StructureCondition(mask)
        self.structure_pixelcnn = StructurePixelcnn()

    def _build(self, sample_shape):
        features1 = tf.random.normal(shape=sample_shape[0])
        features2 = tf.random.normal(shape=sample_shape[1])
        self([features1, features2], training=False)

    def call(self, inputs, training=None, mask=None):
        masked = inputs[0]
        quant_t = inputs[1]
        cond_masked = self.structure_condition(masked)
        pix_out = self.structure_pixelcnn([quant_t, cond_masked])
        return pix_out


if __name__ == '__main__':
    model = Structure(config=None, mask=None)
    model._build([[3, 256, 256, 3], [1, 256, 256, 3]])

