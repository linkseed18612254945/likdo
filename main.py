from train import env
from configs import build_image_caption_config
from models import kernels
from torch import optim
from datasets.image_caption_dataest import get_image_caption_data
from utils.meter import GroupMeters


if __name__ == '__main__':
    config = build_image_caption_config()

    train_dataset, train_loader, valid_dataset, valid_loader = get_image_caption_data(config)
    config.data.vocab_size = len(train_dataset.dataset.vocab)

    model = kernels.ImageCaption(config)
    optimizer = optim.Adam(params=model.parameters(), lr=config.train.lr)
    trainer = env.TrainEnv(model, optimizer, config)

    meters = GroupMeters()
    for epoch in range(1, config.train.epoch_size):
        meters.reset()
        trainer.train_epoch(train_loader, epoch, meters)
