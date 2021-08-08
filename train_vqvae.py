import argparse
from utils.user_config import UserConfig
from dataloaders.vqvae_dataloader import VQVAE_DataLoader
from trainer.vqvae_trainer import Vqvae_Trainer
from model.vqvae import VQVAE
from base_train import Base_Train


class VQVAE_Train(Base_Train):

    def __init__(self, config):
        super(VQVAE_Train, self).__init__(config=config)
        self.config = config
        self.vqvae = VQVAE(config)
        self.data_loader = VQVAE_DataLoader(config)
        self.data_loader.load_state(self.config['learning_config']['running_config']['vqvae_checkpoints_dir'])
        self.runner = Vqvae_Trainer(config['learning_config']['running_config'])
        self.opt = self.load_opt()
        self.runner.set_total_train_steps(
            config['learning_config']['running_config']['max_steps']
        )
        self.runner.compile(self.vqvae, self.opt)
        self.data_loader.batch = self.runner.global_batch_size
        self.training_flag = True


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_config', type=str, default='./config/data_config.yaml',
                       help='the am data config path')
    parse.add_argument('--model_config', type=str, default='./config/model_config.yaml',
                       help='the am model config path')
    args = parse.parse_args()
    config = UserConfig(args.data_config, args.model_config)
    train = VQVAE_Train(config)
    train.train()
