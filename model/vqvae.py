import tensorflow as tf
from tensorflow.python.training import moving_averages


class Resnet(tf.keras.layers.Layer):
    def __init__(self,
                 nr_res_channel=64,
                 nr_channel=128,
                 name='resnet'):
        super(Resnet, self).__init__(name=name)
        self.ac1 = tf.keras.layers.ELU()
        self.conv1 = tf.keras.layers.Conv2D(nr_res_channel, 3, padding='same')
        self.ac2 = tf.keras.layers.ELU()
        self.conv2 = tf.keras.layers.Conv2D(nr_channel, 1, padding='same')

    def call(self, inputs, training=False):
        outputs = self.conv1(self.ac1(inputs))
        outputs = self.conv2(self.ac2(outputs))
        return inputs + outputs


class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self,
                 nr_embedding,
                 emb_dim,
                 beta=0.25,
                 name='VectorQuantizer'):
        super(VectorQuantizer, self).__init__(name=name)
        self.nr_embedding = nr_embedding
        self.emb_dim = emb_dim
        self.beta = beta
        self.ema_cluster_size = nr_embedding
        # self.embedding = tf.keras.layers.Embedding(self.nr_embedding, self.emb_dim)
        self.initializer = 'uniform'
        # self.embedding = tf.Variable(initializer(shape=(self.nr_embedding, self.emb_dim)), name='embedding')

    def build(self, input_shape):
        self.w = self.add_weight(name='embedding', shape=(self.nr_embedding, self.emb_dim),
                                 initializer=self.initializer, trainable=True)
        super(VectorQuantizer, self).build(input_shape)

    def call(self, inputs, is_training=True, **kwargs):
        # inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        inputs_flattened = tf.reshape(inputs, [-1, self.emb_dim])

        distances = (tf.math.reduce_sum(inputs_flattened ** 2, 1, keepdims=True)
                     - 2 * tf.matmul(inputs_flattened, tf.transpose(self.w))
                     + tf.math.reduce_sum(tf.transpose(self.w) ** 2, 0, keepdims=True))
        # distances = 2 * tf.matmul(inputs_flattened, tf.transpose(self.w))

        encodings_indices = tf.argmax(-distances, 1)
        quantized = tf.nn.embedding_lookup(self.w, encodings_indices)
        quantized = tf.reshape(quantized, inputs.shape)
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        loss = self.beta * e_latent_loss
        quantized = inputs + tf.stop_gradient(quantized - inputs)

        return quantized, loss, encodings_indices, tf.transpose(self.w)


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 nr_channel=128,
                 nr_res_block=2,
                 nr_res_channel=64,
                 embedding_dim=64,
                 num_embeddings=512,
                 commitment_cost=0.25,
                 decay=0.99,
                 training=False,
                 name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        # Bottom encoder
        self.conv1 = tf.keras.layers.Conv2D(nr_channel // 2, 4, strides=(2, 2), padding='same')
        self.elu1 = tf.keras.layers.ELU()
        self.conv2 = tf.keras.layers.Conv2D(nr_channel, 4, strides=(2, 2), padding='same')
        self.elu2 = tf.keras.layers.ELU()
        self.conv3 = tf.keras.layers.Conv2D(nr_channel, 3, padding='same')
        self.res_blocks_1 = []
        for i in range(nr_res_block):
            single_res_block = Resnet(nr_res_channel, nr_channel)
            self.res_blocks_1.append(single_res_block)
        self.elu3 = tf.keras.layers.ELU()

        # Top encoder
        self.conv4 = tf.keras.layers.Conv2D(nr_channel // 2, 4, strides=(2, 2), padding='same')
        self.elu4 = tf.keras.layers.ELU()
        # self.conv5 = tf.keras.layers.Conv2D(nr_channel, 4, strides=(2, 2), padding='same')
        self.elu5 = tf.keras.layers.ELU()
        self.conv6 = tf.keras.layers.Conv2D(nr_channel, 3, padding='same')
        self.res_blocks_2 = []
        for i in range(nr_res_block):
            single_res_block = Resnet(nr_res_channel, nr_channel)
            self.res_blocks_2.append(single_res_block)
        self.elu6 = tf.keras.layers.ELU()
        self.conv7 = tf.keras.layers.Conv2D(embedding_dim, 1, padding='same')

        # Vector quantization with top codebook
        self.vector_quantization_top = VectorQuantizer(num_embeddings, embedding_dim)

        # Top decoder
        self.conv_dc_1 = tf.keras.layers.Conv2D(nr_channel, 3, padding='same')
        self.res_blocks_dc = []
        for rep in range(nr_res_block):
            single_res_block = Resnet(nr_res_channel, nr_channel)
            self.res_blocks_dc.append(single_res_block)
        self.elu_dc_1 = tf.keras.layers.ELU()
        self.tp_conv_1 = tf.keras.layers.Conv2DTranspose(nr_channel, 4, strides=(2, 2), padding='same')
        self.conv_dc_2 = tf.keras.layers.Conv2D(embedding_dim, 1, padding='same')

        # Vector quantization with bottom codebook
        self.vector_quantization_bottom = VectorQuantizer(num_embeddings, embedding_dim)

    def call(self, inputs, **kwargs):
        # Bottom encoder
        outputs = self.elu1(self.conv1(inputs))
        outputs = self.elu2(self.conv2(outputs))
        outputs = self.conv3(outputs)
        for res_block in self.res_blocks_1:
            outputs = res_block(outputs)
        enc_b = self.elu3(outputs)

        # Top encoder
        outputs = self.elu4(self.conv4(enc_b))
        outputs = self.conv6(outputs)
        for res_block in self.res_blocks_2:
            outputs = res_block(outputs)
        outputs = self.elu6(outputs)
        enc_t = self.conv7(outputs)

        # Vector quantization with top codebook
        quant_t, diff_t, idx_t, embed_t = self.vector_quantization_top(enc_t)

        # top decoder
        outputs = self.conv_dc_1(quant_t)
        for res_block in self.res_blocks_dc:
            outputs = res_block(outputs)
        outputs = self.elu_dc_1(outputs)
        dec_t = self.tp_conv_1(outputs)
        outputs = tf.concat([enc_b, dec_t], -1)
        enc_b = self.conv_dc_2(outputs)

        # Vector quantization with bottom codebook
        quant_b, diff_b, idx_b, embed_b = self.vector_quantization_bottom(enc_b)
        loss = diff_b + diff_t
        return enc_t, enc_b, quant_t, quant_b, loss, idx_t, idx_b, embed_t, embed_b


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 ema=None,
                 nr_channel=128,
                 nr_res_block=2,
                 nr_res_channel=64,
                 embedding_dim=64,
                 name='Decoder'):
        super(Decoder, self).__init__(name=name)
        # Bottom decoder
        self.tp_conv_1 = tf.keras.layers.Conv2DTranspose(embedding_dim, 4, strides=(2, 2), padding='same')
        self.conv_1 = tf.keras.layers.Conv2D(nr_channel, 3, padding='same')
        self.dec_res_block = []
        for rep in range(nr_res_block):
            single_res_block = Resnet(nr_res_channel)
            self.dec_res_block.append(single_res_block)
        self.elu1 = tf.keras.layers.ELU()
        self.tp_conv_2 = tf.keras.layers.Conv2DTranspose(nr_channel // 2, 4, strides=(2, 2), padding='same')
        self.elu2 = tf.keras.layers.ELU()
        self.tp_conv_3 = tf.keras.layers.Conv2DTranspose(3, 4, strides=(2, 2), padding='same')

    def call(self, inputs, **kwargs):
        quant_t = inputs[0]
        quant_b = inputs[1]
        quant_t = self.tp_conv_1(quant_t)
        output = tf.concat([quant_b, quant_t], -1)
        output = self.conv_1(output)
        for res_block in self.dec_res_block:
            output = res_block(output)
        output = self.elu1(output)
        output = self.tp_conv_2(output)
        output = self.elu2(output)
        dec_b = self.tp_conv_3(output)
        return dec_b


class VQVAE(tf.keras.Model):
    def __init__(self, config, name='VQVAE', **kwargs):
        super(VQVAE, self).__init__(name=name)
        vae_config = config['vqvae']
        self.nr_channel_vq = vae_config['nr_channel_vq']
        self.nr_res_block_vq = vae_config['nr_res_block_vq']
        self.nr_res_channel_vq = vae_config['nr_res_channel_vq']
        self.embedding_dim = vae_config['embedding_dim']
        self.num_embeddings = vae_config['num_embeddings']
        self.commitment_cost = vae_config['commitment_cost']
        self.decay = vae_config['decay']

        self.encoder = Encoder(nr_channel=self.nr_channel_vq, nr_res_channel=self.nr_res_block_vq,
                               nr_res_block=self.nr_res_block_vq, embedding_dim=self.embedding_dim,
                               num_embeddings=self.num_embeddings, commitment_cost=self.commitment_cost,
                               decay=self.decay)
        self.decoder = Decoder(nr_channel=self.nr_channel_vq, nr_res_channel=self.nr_res_channel_vq,
                               nr_res_block=self.nr_res_block_vq, embedding_dim=self.embedding_dim)

    def _build(self, sample_shape):
        features = tf.random.normal(shape=sample_shape)
        self(features, training=False)

    def call(self, inputs, training=None, mask=None):
        enc_t, enc_b, quant_t, quant_b, loss, idx_t, idx_b, embed_t, embed_b = self.encoder(inputs)
        dec_b = self.decoder([quant_t, quant_b])
        return loss, dec_b



