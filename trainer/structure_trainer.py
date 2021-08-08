import logging

import tensorflow as tf
from trainer.base_runners import BaseTrainer


class StructureTrainer(BaseTrainer):
    def __init__(self, config, strategy=None, mask=None, vae_model=None, num_embeddings=None):
        self.outdir = config['sg_ckp_dir']
        self.config = config
        self.set_strategy(strategy)
        self.mask = mask
        self.vae_model = vae_model
        self.num_embeddings = num_embeddings
        super(StructureTrainer, self).__init__(config=config)


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
        _, _, quant_t, _, _, idx_t, _, _, _ = self.vae_model(images)
        encoding_gt = tf.one_hot(idx_t, self.num_embeddings)
        pix_out = self.model([mask, quant_t])
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pix_out, labels=encoding_gt))

        self.train_metrics["total_loss"].update_state(loss)

    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs = epoch
        self._train_batches()
        self._check_eval_interval()

