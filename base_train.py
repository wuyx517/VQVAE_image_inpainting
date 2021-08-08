import os
import tensorflow as tf
import logging


class Base_Train:
    def __init__(self, config):
        self.config = config
        self.data_loader = None
        self.runner = None

    def load_opt(self):
        """Load optimizer from config"""

        opt_list = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']
        opt_name = self.config['optimizer_config']['current_opt']
        if opt_name in opt_list:
            optimizer = eval(f"tf.keras.optimizers.{opt_name}(**self.config['optimizer_config']['{opt_name}'])")
            return optimizer
        else:
            raise ValueError('Unknown optimizer')

    def load_checkpoint(self, config, model):
        """Load checkpoint."""
        self.checkpoint_dir = os.path.join(config['learning_config']['running_config']["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.init_steps = int(files[-1].split('_')[-1].replace('.h5', ''))

    def train(self):
        train_generator = self.data_loader.generator(train=True)
        eval_generator = self.data_loader.generator(train=False)
        train_datasets = tf.data.Dataset.from_generator(lambda: train_generator,
                                                        self.data_loader.return_data_types(),
                                                        self.data_loader.return_data_shape(), )
        eval_datasets = tf.data.Dataset.from_generator(lambda: eval_generator,
                                                       self.data_loader.return_data_types(),
                                                       self.data_loader.return_data_shape())
        self.runner.set_datasets(train_datasets, eval_datasets)

        # If load finished model, skip training
        if self.runner._finished():
            logging.info('Finish training!')
            self.training_flag = False

        while self.training_flag:
            self.runner.fit(epoch=self.data_loader.epochs)

            if self.runner._finished():
                self.runner.save_checkpoint()
                logging.info('Finish training!')
                break

            if self.runner.steps % self.config['learning_config']['running_config']['save_interval_steps'] == 0:
                self.data_loader.save_state(self.config['learning_config']['running_config']['outdir'])
