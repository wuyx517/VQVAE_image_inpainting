import tensorflow as tf
from model.layer import GatedConv2d, attention_transfer, GatedDeconv2d


class TextureGeneratorSpec(tf.keras.layers.Layer):
    def __init__(self, nr_channel=64, **kwargs):
        super().__init__(**kwargs)
        self.nr_channel = nr_channel
        # Encoder
        self.rt_conv_1 = GatedConv2d(num_filters=nr_channel, filter_size=(5, 5), stride=(1, 1))
        self.rt_conv_2 = GatedConv2d(num_filters=nr_channel*2, filter_size=(3, 3), stride=(2, 2))
        self.rt_conv_3 = GatedConv2d(num_filters=nr_channel*3, filter_size=(3, 3), stride=(2, 2))
        self.rt_conv_4 = GatedConv2d(num_filters=nr_channel*4, filter_size=(3, 3), stride=(2, 2))
        # Upsample structure feature maps (with quantization)
        self.rt_conv_5 = GatedConv2d(num_filters=nr_channel * 4, filter_size=(3, 3), stride=(1, 1))
        self.rt_conv_6 = GatedConv2d(num_filters=nr_channel * 4, filter_size=(3, 3), stride=(1, 1))
        self.rt_conv_7 = GatedDeconv2d(num_filters=nr_channel * 4)
        self.rt_conv_8 = GatedConv2d(num_filters=nr_channel * 4, filter_size=(3, 3), stride=(1, 1))
        self.rt_conv_9 = GatedConv2d(num_filters=nr_channel * 4, filter_size=(3, 3), stride=(1, 1))
        self.rt_conv_10 = GatedConv2d(num_filters=nr_channel * 4, filter_size=(3, 3), stride=(1, 1), rate=2)
        self.rt_conv_11 = GatedConv2d(num_filters=nr_channel * 4, filter_size=(3, 3), stride=(1, 1), rate=4)
        self.rt_conv_12 = GatedConv2d(num_filters=nr_channel * 4, filter_size=(3, 3), stride=(1, 1), rate=8)
        self.rt_conv_13 = GatedConv2d(num_filters=nr_channel * 4, filter_size=(3, 3), stride=(1, 1), rate=16)
        # Attention transfer under the guidance of structure feature maps
        # self.attention_transfer = AttentionTransfer(ksize=3, stride=1, fuse_k=3, softmax_scale=50., fuse=True)
        # Decoder
        self.rt_conv_14 = GatedConv2d(num_filters=nr_channel * 4, filter_size=(3, 3), stride=(1, 1))
        self.rt_conv_15 = GatedConv2d(num_filters=nr_channel * 4, filter_size=(3, 3), stride=(1, 1))
        self.rt_conv_16 = GatedConv2d(num_filters=nr_channel * 4, filter_size=(3, 3), stride=(1, 1))
        self.rt_deconv_1 = GatedDeconv2d(num_filters=nr_channel * 2)
        self.rt_conv_17 = GatedConv2d(num_filters=nr_channel * 2, filter_size=(3, 3), stride=(1, 1))
        self.rt_deconv_2 = GatedDeconv2d(num_filters=nr_channel)
        self.rt_conv_18 = GatedConv2d(num_filters=nr_channel, filter_size=(3, 3), stride=(1, 1))
        self.conv = tf.keras.layers.Conv2D(3, (3, 3), strides=(1, 1), activation='tanh')

    def _build(self, sample_shape):
        features1 = tf.random.normal(shape=sample_shape[0])
        features2 = tf.random.normal(shape=sample_shape[1])
        features3 = tf.random.normal(shape=sample_shape[2])
        self([features1, features2, features3], training=False)

    def call(self, inputs, *args, **kwargs):
        x = inputs[0]
        mask = inputs[1]
        s = inputs[2]
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x_in = tf.concat([x, ones_x, ones_x * mask], axis=3)

        # Encoder
        pl1 = self.rt_conv_1(x_in)
        pl1 = self.rt_conv_2(pl1)
        pl1 = self.rt_conv_3(pl1)
        pl2 = self.rt_conv_4(pl1)
        # Upsample structure feature maps (with quantization)
        x_s = self.rt_conv_5(s)
        x_s = self.rt_conv_6(x_s)
        x_s = self.rt_conv_7(x_s)
        pl2 = tf.concat([pl2, x_s], axis=-1)
        pl2 = self.rt_conv_8(pl2)
        pl2 = self.rt_conv_9(pl2)
        pl2 = self.rt_conv_10(pl2)
        pl2 = self.rt_conv_11(pl2)
        pl2 = self.rt_conv_12(pl2)
        pl2 = self.rt_conv_13(pl2)
        # Attention transfer under the guidance of structure feature maps
        pl1_att, pl2_att = attention_transfer(s, pl1, pl2, 3, 1, 3, softmax_scale=50., fuse=True)

        # Decoder
        x = self.rt_conv_14(pl2)
        x = self.rt_conv_15(x)
        x = tf.concat([x, pl2_att], axis=-1)
        x = self.rt_conv_16(x)
        x = self.rt_deconv_1(x)
        x = tf.concat([x, pl1_att], axis=-1)
        x = self.rt_conv_17(x)
        x = self.rt_deconv_2(x)
        x = self.rt_conv_18(x)
        x = self.conv(x)

        return x


class TextureDiscriminatorSpec(tf.keras.layers.Layer):
    def __init__(self, nr_channel=64, name='TextureDiscriminatorSpec', **kwargs):
        super().__init__(name=name, **kwargs)
        nr_channel_list = [1, 2, 4, 4, 4, 4]
        self.snconv_list = []
        for nr in nr_channel_list:
            single_conv = tf.keras.layers.Conv2D(nr_channel * nr, 5, strides=(2, 2), activation='leaky_relu',
                                                 kernel_initializer='TruncatedNormal')
            self.snconv_list.append(single_conv)
        self.flatten = tf.keras.layers.Flatten()

    def _build(self, sample_shape):
        features1 = tf.random.normal(shape=sample_shape)
        self(features1, training=False)

    def call(self, inputs, *args, **kwargs):
        output = inputs
        for single_conv in self.snconv_list:
            output = single_conv(output)
        output = self.flatten(output)
        return output


# class Texture(tf.keras.Model):
#     def __init__(self, config, name='Texture', *args, **kwargs):
#         super().__init__(name=name, *args, **kwargs)
#         self.config = config
#         self.batch = config['learning_config']['running_config']['batch_size']
#         self.gexture_generator_spec = TextureGeneratorSpec()
#         self.texture_discriminator_spec = TextureDiscriminatorSpec()
#
#     def call(self, inputs, training=None, mask=None):
#         (masked, mask, quant_t, img) = inputs
#         gen_out = self.gexture_generator_spec(inputs[:3])
#         batch_complete = gen_out * mask + masked * (1. - mask)
#         pos_neg = tf.concat([img, batch_complete], axis=0)
#         pos_neg = tf.concat([pos_neg, tf.tile(mask, [self.batch * 2, 1, 1, 1])], axis=3)
#         output = self.texture_discriminator_spec(pos_neg)
#         return gen_out, output, batch_complete






