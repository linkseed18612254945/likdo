from train import env
from configs import build_image_caption_config
from models import image_caption_kernels
from torch import optim
from datasets.image_caption_dataest import get_image_caption_data
from utils.meter import GroupMeters
from utils.cuda import get_gpu_device


if __name__ == '__main__':
    config = build_image_caption_config()
    device = get_gpu_device(config.train.use_gpu, config.train.gpu_index)

    train_dataset, train_loader, valid_dataset, valid_loader = get_image_caption_data(config)
    config.data.vocab_size = len(train_dataset.dataset.vocab)

    model = image_caption_kernels.BasicImageCaption(config).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=config.train.lr)
    trainer = env.TrainEnv(model, optimizer, config, device)

    meters = GroupMeters()
    for epoch in range(1, config.train.epoch_size + 1):
        meters.reset()
        trainer.train_epoch(train_loader, epoch, meters)
        trainer.save_checkpoint(config.train.save_model_path)

