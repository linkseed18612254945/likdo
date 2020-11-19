from utils import env
import configs
from models import image_caption_kernels, vqa_kernels, text_classify_kernels
from torch import optim
from datasets.image_caption_dataest import *
from datasets.vqa_dataset import *
from datasets.image_classify_dataset import get_classify_data
from datasets.text_classify_dataset import get_text_classify_data
from utils.meter import GroupMeters
from utils.cuda import get_gpu_device
from utils.logger import get_logger
from eval import text_classify_meters


logger = get_logger(__file__)

def image_caption_main():
    logger.critical("Build train config and device")
    config = configs.build_image_caption_config()
    device = get_gpu_device(config.train.use_gpu, config.train.gpu_index)

    logger.critical("Build train and validation data")
    train_dataset, train_loader, valid_dataset, valid_loader = get_image_caption_data(config)
    config.data.caption_vocab_size = len(train_dataset.vocab)

    logger.critical("Init model and train environment")
    # model = image_caption_kernels.BasicImageCaption(config).to(device)
    model = image_caption_kernels.IMAGINET(config).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=config.train.lr)
    trainer = env.TrainEnv(model, optimizer, config, device)

    logger.critical("Start to train")
    meters = GroupMeters()
    for epoch in range(1, config.train.epoch_size + 1):
        meters.reset()
        trainer.train_epoch(train_loader, epoch, meters)
        trainer.save_checkpoint(config.train.save_model_path)


def vqa_main():
    logger.critical("Build VQA train config and device")
    config = configs.build_vqa_config()
    device = get_gpu_device(config.train.use_gpu, config.train.gpu_index)

    logger.critical("Build VQA train and validation data")
    train_dataset, train_loader, valid_dataset, valid_loader = get_vqa_dataset(config)
    config.data.question_vocab_size = len(train_dataset.question_vocab)
    config.data.answer_vocab_size = len(train_dataset.answer_vocab)

    logger.critical("Init VQA model and train environment")
    model = vqa_kernels.OnlyText(config).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=config.train.lr)
    trainer = env.TrainEnv(model, optimizer, config, device)

    logger.critical("Start to train")
    meters = GroupMeters()
    for epoch in range(1, config.train.epoch_size + 1):
        meters.reset()
        trainer.train_epoch(train_loader, epoch, meters)
        trainer.save_checkpoint(config.train.save_model_path)


def image_classify_main():
    logger.critical("Build train config and device")
    config = configs.build_image_classify_config()
    device = get_gpu_device(config.train.use_gpu, config.train.gpu_index)

    logger.critical("Build train and validation data")
    train_dataset, train_loader, valid_dataset, valid_loader = get_classify_data(config)
    config.data.class_nums = len(train_dataset.get_class_map())

    logger.critical("Init model and train environment")
    model = image_caption_kernels.IMAGINET(config).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=config.train.lr)
    trainer = env.TrainEnv(model, optimizer, config, device)

    logger.critical("Start to train")
    meters = GroupMeters()
    for epoch in range(1, config.train.epoch_size + 1):
        meters.reset()
        trainer.train_epoch(train_loader, epoch, meters)
        trainer.save_checkpoint(config.train.save_model_path)

def text_classify_main():
    logger.critical("Build train config and device")
    config = configs.build_text_classify_config()
    device = get_gpu_device(config.train.use_gpu, config.train.gpu_index)

    logger.critical("Build train and validation data")
    train_dataset, train_loader, valid_dataset, valid_loader = get_text_classify_data(config)
    config.data.vocab_size = len(train_dataset.vocab)
    config.data.num_labels = train_dataset.num_labels

    logger.critical("Init model and train environment")
    model = text_classify_kernels.BertBaseline(config).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=config.train.lr)
    trainer = env.TrainEnv(model, optimizer, config, device)

    logger.critical("Start to train")
    meters = GroupMeters()
    valid_meter = text_classify_meters.DBpediaConceptMeter()
    for epoch in range(1, config.train.epoch_size + 1):
        meters.reset()
        trainer.train_epoch(train_loader, epoch, meters)
        trainer.save_checkpoint(config.train.save_model_path)

        meters.reset()
        trainer.valid_epoch(valid_loader, epoch, valid_meter)


if __name__ == '__main__':
    text_classify_main()


