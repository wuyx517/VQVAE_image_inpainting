import logging
import os
import numpy as np
import tensorflow as tf
from trainer.base_runners import BaseTrainer
import cv2

class StructureTrainer(BaseTrainer):
    def __init__(self, config, strategy=None, mask=None, vae_model=None, num_embeddings=None):
        self.outdir = config['sg_ckp_dir']
        self.config = config['learning_config']['running_config']
        self.global_config = config
        self.set_strategy(strategy)
        self.mask = mask
        self.vae_model = vae_model
        self.num_embeddings = num_embeddings
        self.gap = 2
        self.image_size = self.global_config['data_config']['image_size']
        super(StructureTrainer, self).__init__(config=config)
        self.checkpoint_dir = os.path.join(self.config["sg_ckp_dir"], "texture_ckp_dir")


    def set_train_metrics(self):
        self.train_metrics = {
            'total_loss': tf.keras.metrics.Mean('train_total_loss', dtype=tf.float32),
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            'total_loss': tf.keras.metrics.Mean('eval_total_loss', dtype=tf.float32),
        }


    def compile(self, model, optimizer):
        self.model = model
        self.model._build([[1, 64, 64, 3], [1, 8, 8, 64]])
        try:
            self.load_checkpoint()
        except:
            logging.info('trainer resume failed')
        self.model.summary(line_length=100)
        self.set_progbar()
        self.optimizer = optimizer

    # @tf.function()
    def _train_step(self, batch):
        images = batch
        mask = images * (1. - self.mask)
        _, _, quant_t, _, _, idx_t, _, _, _ = self.vae_model(images)
        encoding_gt = tf.one_hot(idx_t, self.num_embeddings)
        with tf.GradientTape() as tape:
            pix_out = self.model([mask, quant_t])
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pix_out, labels=encoding_gt))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients((grad, var)
                                           for (grad, var) in zip(gradients, self.model.trainable_variables)
                                           if grad is not None)
            self.train_metrics["total_loss"].update_state(loss)

    def _test_step(self, batch):
        images = batch
        mask = images * (1. - self.mask)
        enc_t, enc_b, quant_t, quant_b, loss, idx_t, idx_b, embed_t, embed_b = self.vae_model(images)
        encoding_gt = tf.one_hot(idx_t, self.num_embeddings)
        pix_out = self.model([mask, quant_t])
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pix_out, labels=encoding_gt))

        self.train_metrics["total_loss"].update_state(loss)

    def _generate_step(self, image):
        image = tf.expand_dims(image, axis=0)

        embdding_dim = int(self.global_config['vqvae']['embedding_dim'])
        num_embdding = int(self.global_config['vqvae']['num_embeddings'])
        top_shape = (self.image_size // 8, self.image_size // 8, 1)
        masked = image * (1. - self.mask)
        enc_t, enc_b, quant_t, quant_b, loss, idx_t, idx_b, embed_t, embed_b = self.vae_model.Encoder(image)
        dec_b = self.vae_model.Decoder(image)
        cond_masked = self.model.StructureCondition(masked)
        recons_gt = tf.clip_by_value(dec_b, -1, 1)
        e_gen = np.zeros((1, self.image_size // 8, self.image_size // 8, embdding_dim), dtype=np.float32)
        pix_out = self.model.StructurePixelcnn(e_gen, cond_masked, dropout_p=0)
        pix_out = tf.reshape(pix_out, (-1, num_embdding))
        probs_out = tf.nn.log_softmax(pix_out, axis=-1)
        samples_out = tf.random.categorical(probs_out, 1)
        samples_out = tf.reshape(samples_out, (-1,) + top_shape[:-1])
        new_e_gen = tf.nn.embedding_lookup(tf.transpose(embed_t, [1, 0]), samples_out, validate_indices=False)

        for yi in range(top_shape[0]):
            for xi in range(top_shape[1]):
                e_gen[:, yi, xi, :] = new_e_gen[:, yi, xi, :]

        recons_gen = tf.clip_by_value(dec_b, -1, 1)
        return image, masked, recons_gen, recons_gen

    def _save_generate_image(self, image):
        image, masked, recons_gen, recons_gen = self._generate_step(image)
        height = self.image_size * 2 + self.gap * 3

        result = 255 * np.ones((height, height, 3), dtype=np.uint8)
        result[: self.image_size, : self.image_size] = image
        result[: self.image_size, (self.image_size + self.gap):] = masked
        result[(self.image_size + self.gap):, self.image_size] = recons_gen
        result[(self.image_size + self.gap):, (self.image_size + self.gap):] = recons_gen
        cv2.imwrite(self.config['sg_ckp_dir'])

    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs = epoch
        self._train_batches()
        self._check_eval_interval()
