import numpy as np
import random
import tensorflow as tf
import os


class VQVAE_DataLoader:
    def __init__(self, config, mask=None):
        self.load_size = config['data_config']['load_size']
        self.image_size = config['data_config']['image_size']
        self.train_list = self.make_file_list(config['data_config']['train_flist'])
        self.pick_index = [0.] * len(self.train_list)
        self.eval_list = self.make_file_list(config['data_config']['valid_flist'])
        self.epochs = 1
        self.batch = config['learning_config']['running_config']['batch_size']
        self.len_data = len(self.train_list)
        self.mask = mask
        self.is_mask = False

    def make_file_list(self, flist):
        if not isinstance(flist, list):
            flist = [flist]
        data = []
        for file in flist:
            with open(file, encoding='utf-8') as f:
                data_txt = f.readlines()
            data.extend([i.strip() for i in data_txt if i != ''])
        return data

    def load_state(self, outdir):
        try:
            self.pick_index = np.load(os.path.join(outdir, 'dg_state.npy')).flatten().tolist()
            self.epochs = 1 + int(np.mean(self.pick_index))
        except FileNotFoundError:
            print('not found state file')
        except:
            print('load state falied,use init state')

    def save_state(self, outdir):
        np.save(os.path.join(outdir, 'dg_state.npy'), np.array(self.pick_index))

    def return_data_types(self):
        return (tf.float32)

    def return_data_shape(self):
        return ([None, None, None, 3])

    def get_per_epoch_steps(self):
        return len(self.train_list) // self.batch

    def generate(self, train=True):
        if train:
            batch = self.batch
            indexs = np.argsort(self.pick_index)[:batch]
            indexs = random.sample(indexs.tolist(), batch // 2)
            sample = [self.train_list[i] for i in indexs]
            for i in indexs:
                self.pick_index[int(i)] += 1
            self.epochs = 1 + int(np.mean(self.pick_index))
        else:
            sample = random.sample(self.eval_list, self.batch)

        img_list = []
        for img_path in sample:
            img_file = tf.io.read_file(img_path)
            img_decoded = tf.cond(tf.image.is_jpeg(img_file),
                                  lambda: tf.image.decode_jpeg(img_file, channels=3),
                                  lambda: tf.image.decode_png(img_file, channels=3))
            img = tf.cast(img_decoded, tf.float32)
            # print(img.shape)
            if train:
                img = tf.image.resize(img, [self.load_size, self.load_size])
                img = tf.image.random_crop(img, [self.image_size, self.image_size, 3])
            else:
                img = tf.image.resize(img, [self.image_size, self.image_size])

            img = tf.clip_by_value(img, 0., 255.)
            img = img / 127.5 - 1
            img_list.append(img)
        img = np.array(img_list, 'float32')
        # print(img.shape)
        return img

    def generator(self, train=True):
        while True:
            img = self.generate(train=train)
            yield img







