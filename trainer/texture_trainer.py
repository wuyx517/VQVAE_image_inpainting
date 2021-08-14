import logging
import os
import tensorflow as tf
from trainer.base_runners import BaseTrainer
from model.layer import feature_loss


class TextureTrainer(BaseTrainer):
    def __init__(self, config, strategy=None, mask=None, vae_model=None, num_embeddings=None, fea_loss_weight=0.1):
        self.outdir = config['sg_ckp_dir']
        self.config = config
        self.set_strategy(strategy)
        self.mask = mask
        self.vae_model = vae_model
        self.num_embeddings = num_embeddings
        self.fea_loss_weight = fea_loss_weight
        super(TextureTrainer, self).__init__(config=config)

    def set_train_metrics(self):
        self.train_metrics = {
            'gen_loss': tf.keras.metrics.Mean('gen_loss', dtype=tf.float32),
            'dis_loss': tf.keras.metrics.Mean('dis_loss', dtype=tf.float32)
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            'total_loss': tf.keras.metrics.Mean('eval_total_loss', dtype=tf.float32),
        }


    def compile(self, gen, dis, optimizer):
        self.gen = gen
        self.gen._build([[2, 256, 256, 3], [2, 256, 256, 1], [2, 32, 32, 64]])
        self.dis = dis
        self.dis._build(sample_shape=[2, 64, 64, 4])
        try:
            self.load_checkpoint()
        except:
            logging.info('trainer resume failed')
        print('gen summary is : ')
        self.gen.summary(line_length=100)
        print('dis summary is : ')
        self.dis.summary(line_length=100)

        self.set_progbar()
        self.optimizer = optimizer

    def load_checkpoint(self, ):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints", "gen")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.gen.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.steps = int(files[-1].split('_')[-1].replace('.h5', ''))

        self.checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints", "dis")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.dis.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.steps = int(files[-1].split('_')[-1].replace('.h5', ''))

    # @tf.function()
    def _train_step(self, batch):
        images = batch
        masked = images * (1. - self.mask)
        _, _, quant_t, _, _, idx_t, _, _, _ = self.vae_model(images)
        # encoding_gt = tf.one_hot(idx_t, self.num_embeddings)
        with tf.GradientTape() as tape:
            # gen out
            gen_out = self.gen([masked, self.mask, quant_t])
            # dis out
            batch_complete = gen_out * self.mask + masked * (1. - self.mask)
            pos_neg = tf.concat([images, batch_complete], axis=0)
            pos_neg = tf.concat([pos_neg, tf.tile(self.mask, [images.shape[0] * 2, 1, 1, 1])], axis=3)
            dis_out = self.dis(pos_neg)
            # loss
            ae_loss = tf.reduce_mean(tf.abs(images - gen_out))
            dis_pos, dis_neg = tf.split(dis_out, 2)
            gen_loss = - tf.reduce_mean(dis_neg)
            enc_t, enc_b, quant_t, quant_b, loss, idx_t, idx_b, embed_t, embed_b = self.vae_model(batch_complete)
            fea_loss1 = self.fea_loss_weight * feature_loss(enc_t, idx_t, embed_t)
            fea_loss2 = self.fea_loss_weight * feature_loss(enc_b, idx_b, embed_b)
            fea_loss = fea_loss1 + fea_loss2
            gen_loss = gen_loss + ae_loss + fea_loss

            gradients_gen = tape.gradient(gen_loss, self.gen.trainable_variables)
            self.optimizer.apply_gradients((grad, var)
                                           for (grad, var) in zip(gradients_gen, self.gen.trainable_variables)
                                           if grad is not None)

            self.train_metrics["gen_loss"].update_state(gen_loss)


        with tf.GradientTape() as dis_tape:
            # gen out
            gen_out = self.gen([masked, self.mask, quant_t])
            # dis out
            batch_complete = gen_out * self.mask + masked * (1. - self.mask)
            pos_neg = tf.concat([images, batch_complete], axis=0)
            pos_neg = tf.concat([pos_neg, tf.tile(self.mask, [images.shape[0] * 2, 1, 1, 1])], axis=3)
            dis_out = self.dis(pos_neg)
            # loss
            dis_pos, dis_neg = tf.split(dis_out, 2)
            hinge_pos = tf.reduce_mean(tf.nn.relu(1 - dis_pos))
            hinge_neg = tf.reduce_mean(tf.nn.relu(1 + dis_neg))
            dis_loss = tf.add(.5 * hinge_pos, .5 * hinge_neg)

            gradients_dis = dis_tape.gradient(dis_loss, self.dis.trainable_variables)

            self.optimizer.apply_gradients((grad, var)
                                           for (grad, var) in zip(gradients_dis, self.dis.trainable_variables)
                                           if grad is not None)

            self.train_metrics["dis_loss"].update_state(dis_loss)


    def _test_step(self, batch):
        images = batch
        masked = images * (1. - self.mask)
        _, _, quant_t, _, _, idx_t, _, _, _ = self.vae_model(images)
        encoding_gt = tf.one_hot(idx_t, self.num_embeddings)

        # gen out
        gen_out = self.gen([masked, self.mask, quant_t])
        # dis out
        batch_complete = gen_out * self.mask + masked * (1. - self.mask)
        pos_neg = tf.concat([images, batch_complete], axis=0)
        pos_neg = tf.concat([pos_neg, tf.tile(self.mask, [images.shape[0] * 2, 1, 1, 1])], axis=3)
        dis_out = self.dis(pos_neg)
        # loss
        ae_loss = tf.reduce_mean(tf.abs(images - gen_out))
        dis_pos, dis_neg = tf.split(dis_out, 2)
        gen_loss = - tf.reduce_mean(dis_neg)
        enc_t, enc_b, quant_t, quant_b, loss, idx_t, idx_b, embed_t, embed_b = self.vae_model(batch_complete)
        fea_loss1 = self.fea_loss_weight * feature_loss(enc_t, idx_t, embed_t)
        fea_loss2 = self.fea_loss_weight * feature_loss(enc_b, idx_b, embed_b)
        fea_loss = fea_loss1 + fea_loss2
        gen_loss = gen_loss + ae_loss + fea_loss

        dis_pos, dis_neg = tf.split(dis_out, 2)
        hinge_pos = tf.reduce_mean(tf.nn.relu(1 - dis_pos))
        hinge_neg = tf.reduce_mean(tf.nn.relu(1 + dis_neg))
        dis_loss = tf.add(.5 * hinge_pos, .5 * hinge_neg)

        self.eval_metrics["gen_loss"].update_state(gen_loss)
        self.eval_metrics["dis_loss"].update_state(dis_loss)

    def fit(self, epoch=None):
        if epoch is not None:
            self.epochs = epoch
        self._train_batches()
        self._check_eval_interval()

