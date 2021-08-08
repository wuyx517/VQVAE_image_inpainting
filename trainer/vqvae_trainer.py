import logging

import tensorflow as tf
from trainer.base_runners import BaseTrainer


class Vqvae_Trainer(BaseTrainer):
    def __init__(self, config, strategy=None):
        self.outdir = config['vqvae_checkpoints_dir']
        self.config = config
        self.set_strategy(strategy)
        super(Vqvae_Trainer, self).__init__(config=config)


    def set_train_metrics(self):
        self.train_metrics = {
            'total_loss': tf.keras.metrics.Mean('train_total_loss', dtype=tf.float32),
            'recons_loss': tf.keras.metrics.Mean('train_recons_loss', dtype=tf.float32),
            'commit_loss': tf.keras.metrics.Mean('train_commit_loss', dtype=tf.float32)
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            'total_loss': tf.keras.metrics.Mean('eval_total_loss', dtype=tf.float32),
            'recons_loss': tf.keras.metrics.Mean('eval_recons_loss', dtype=tf.float32),
            'commit_loss': tf.keras.metrics.Mean('eval_commit_loss', dtype=tf.float32)
        }


    def compile(self, model, optimizer):
        self.model = model
        self.model._build([3, 256, 256, 3])
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
        with tf.GradientTape() as tape:
            commit_loss, dec_b = self.model(images)
            recons_loss = tf.reduce_mean((images - dec_b) ** 2)
            loss = commit_loss + recons_loss
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients((grad, var)
                                           for (grad, var) in zip(gradients, self.model.trainable_variables)
                                           if grad is not None)

            self.train_metrics["total_loss"].update_state(loss)
            self.train_metrics["recons_loss"].update_state(recons_loss)
            self.train_metrics["commit_loss"].update_state(commit_loss)

    def _test_step(self, batch):
        images = batch
        commit_loss, dec_b = self.model(images)
        recons_loss = tf.reduce_mean((images - dec_b) ** 2)
        loss = commit_loss + recons_loss

        self.train_metrics["total_loss"].update_state(loss)
        self.train_metrics["recons_loss"].update_state(recons_loss)
        self.train_metrics["commit_loss"].update_state(commit_loss)


    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs = epoch
        self._train_batches()
        self._check_eval_interval()

