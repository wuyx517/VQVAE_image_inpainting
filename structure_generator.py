import argparse
import os
from utils.user_config import UserConfig
from dataloaders.vqvae_dataloader import VQVAE_DataLoader
from trainer.structure_trainer import StructureTrainer
from model.structure_condition import Structure
from model.vqvae import VQVAE
from base_train import Base_Train
from utils.tools import random_bbox, bbox2mask


def load_checkpoint(model, checkpoint_dir):
    """Load checkpoint."""
    checkpoint_dir = os.path.join(checkpoint_dir, 'checkpoints')
    files = os.listdir(checkpoint_dir)
    files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
    model.load_weights(os.path.join(checkpoint_dir, files[-1]))
    return model


class StructureGeneratorTrain(Base_Train):

    def __init__(self, config):
        super(StructureGeneratorTrain, self).__init__(config=config)
        bbox = random_bbox(config['data_config']['image_size'], config['data_config']['image_size'],
                           config['data_config']['margins'], config['data_config']['mask_size'], random_mask=False)
        regular_mask = bbox2mask(bbox, config['data_config']['image_size'], config['data_config']['image_size'],
                                 config['data_config']['max_delta'], name='mask_c')
        mask = regular_mask
        self.config = config
        self.model = Structure(config, mask=mask)
        self.data_loader = VQVAE_DataLoader(config, mask=mask)
        self.data_loader.is_mask = True
        self.data_loader.load_state(self.config['learning_config']['running_config']['vqvae_checkpoints_dir'])
        # load vqvae model
        vae_model = VQVAE(config)
        vae_checkpoint_dir = config['learning_config']['running_config']['vqvae_checkpoints_dir']
        vae_model._build([1, 64, 64, 3])
        vae_model = load_checkpoint(vae_model, vae_checkpoint_dir)
        vae_encoder = vae_model.encoder
        self.runner = StructureTrainer(config, mask=mask, vae_model=vae_encoder,
                                       num_embeddings=config['vqvae']['num_embeddings'])
        self.opt = self.load_opt()
        self.runner.set_total_train_steps(
            config['learning_config']['running_config']['max_steps']
        )
        self.runner.compile(self.model, self.opt)
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
    train = StructureGeneratorTrain(config)
    train.train()