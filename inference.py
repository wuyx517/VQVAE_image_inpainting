from model.vqvae import VQVAE
from model.texture_generator import TextureGeneratorSpec, TextureDiscriminatorSpec
from model.structure_condition import Structure
import os
from utils.tools import random_bbox, bbox2mask
import cv2
import numpy as np
import tensorflow as tf
import sys


def load_checkpoint(model, checkpoint_dir):
    """Load checkpoint."""
    checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
    files = os.listdir(checkpoint_dir)
    files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
    model.load_weights(os.path.join(checkpoint_dir, files[-1]))
    return model


class Inference:
    def __init__(self, image, config):
        super(Inference, self).__init__(config=config)
        bbox = random_bbox(config['data_config']['image_size'], config['data_config']['image_size'],
                           config['data_config']['margins'], config['data_config']['mask_size'], random_mask=False)
        regular_mask = bbox2mask(bbox, config['data_config']['image_size'], config['data_config']['image_size'],
                                 config['data_config']['max_delta'], name='mask_c')
        self.mask = regular_mask
        self.config = config
        self.image_size = self.config['data_config']['image_size']
        self.embedding_dim = self.config['vqvae']['embedding_dim']
        self.num_embedding = self.config['vqvae']['num_embeddings']
        self.top_shape = (self.image_size // 8, self.image_size // 8, 1)

        self.vae_model = VQVAE(config)
        vae_checkpoint_dir = config['learning_config']['running_config']['vqvae_checkpoints_dir']
        self.vae_model._build([1, 64, 64, 3])
        self.vae_model = load_checkpoint(self.vae_model, vae_checkpoint_dir)

        self.structure_model = Structure(config, self.mask)
        sg_ckp_dir = config['learning_config']['running_config']['vqvae_checkpoints_dir']
        self.structure_model._build([1, 64, 64, 3])
        self.structure_model = load_checkpoint(self.structure_model, sg_ckp_dir)

        self.tgs = TextureGeneratorSpec()
        self.tds = TextureDiscriminatorSpec()
        gen_ckp_dir = config['learning_config']['running_config']['texture_gen_ckp_dir']
        self.tgs._build([[2, 256, 256, 3], [2, 256, 256, 1], [2, 32, 32, 64]])
        self.tgs = load_checkpoint(self.tgs, gen_ckp_dir)

        dis_ckp_dir = config['learning_config']['running_config']['texture_dis_ckp_dir']
        self.tds._build([[2, 256, 256, 3], [2, 256, 256, 1], [2, 32, 32, 64]])
        self.tds = load_checkpoint(self.tds, dis_ckp_dir)

    def inference(self, image):
        img_np = cv2.imread(image)[:, :, ::-1].astype(np.float)
        img_np = cv2.resize(img_np, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_np = np.expand_dims(self.mask, -1)

        # Normalize and reshape the image and mask
        img_np = img_np / 127.5 - 1
        mask_np = mask_np / 255.
        img_np = np.expand_dims(img_np, 0)
        mask_np = np.expand_dims(mask_np, 0)
        masked = img_np * (1. - mask_np)

        enc_t, enc_b, quant_t, quant_b, loss, idx_t, idx_b, embed_t, embed_b = self.vae_model.encoder(img_np)
        cond_masked_np = self.structure_model.structure_condition(img_np, mask_np)
        e_gen = np.zeros((1, self.image_size // 8, self.image_size // 8, self.embedding_dim), dtype=np.float32)
        for yi in range(self.top_shape[0]):
            for xi in range(self.top_shape[1]):
                pix_out = self.structure_model.structure_pixelcnn(e_gen, cond_masked_np, dropout_p=0.)
                pix_out = tf.reshape(pix_out, (-1, self.num_embedding))
                probs_out = tf.nn.log_softmax(pix_out, axis=-1)
                samples_out = tf.random.categorical(probs_out, 1)
                samples_out = tf.reshape(samples_out, (-1,) + self.top_shape[:-1])
                new_e_gen = tf.nn.embedding_lookup(tf.transpose(embed_t, [1, 0]), samples_out,
                                                   validate_indices=False)
                e_gen[:, yi, xi, :] = new_e_gen[:, yi, xi, :]
        gen_out = self.tgs(masked, mask_np, e_gen)
        img_gen = gen_out * mask_np + masked * (1. - mask_np)
        output = ((img_gen[0] + 1.) * 127.5).astype(np.uint8)
        cv2.imwrite(os.path.join('save_image', 'inference.img'), output[:, :, ::-1])
        print('inference.png is generated.')
        sys.stdout.flush()
