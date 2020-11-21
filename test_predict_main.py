import tqdm

from datasets.text_classify_dataset import get_test_data
import configs
from utils.cuda import get_gpu_device
from utils import text_utils
from utils.logger import get_logger
from datasets.common.vocab import Vocab
from transformers import pipeline

logger = get_logger(__file__)

def text_classify_main():
    logger.critical("Build test/predict config and device")
    config = configs.build_text_classify_config()
    device = get_gpu_device(config.train.use_gpu, config.train.gpu_index)

    # For zero shot
    labels = Vocab.from_json(config.data.label_vocab_path).idx2word
    config.data.test_use_labels = labels[:5]
    logger.critical(f"Predict_labels: {' '.join(labels)}")

    logger.critical("Build test data")
    test_dataset = get_test_data(config, text_transformer=lambda x: x,
                                 label_name_transformer=lambda x: x, collate_fn=None)

    logger.critical("Init pipline")
    predictor = pipeline("zero-shot-classification", model=config.predict.model_path, device=device)
    # pbar = tqdm.tqdm(test_dataset)
    # for feed_dict in pbar:
    #     feed_dict = {k: v.to(device) if 'to' in v.__dir__() else v for k, v in feed_dict.items()}
    #     output_dict = predictor(feed_dict['text'], )
    #     pbar.set_description(desc=f'Predicting', refresh=True)

if __name__ == '__main__':
    text_classify_main()