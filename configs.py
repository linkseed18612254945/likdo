from utils.container import BasicConfig
import utils

def build_vqa_config():
    config = BasicConfig()

    # Data config setting
    config.data.name = 'cifar10'
    config.data.train_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_train_questions.json'
    config.data.valid_path = None
    config.data.train_data_nums = 'all'
    config.data.valid_data_nums = 'all'

    config.data.classes_nums = None

    # Model config setting
    config.model.image_embedding_dim = 224
    config.model.question_embedding_dim = 224
    config.model.question_rnn_hidden_size = 256
    config.model.question_rnn_num_layers = 1
    config.model.question_rnn_is_bi = False
    config.model.rnn_dropout_rate = 0.5

    # Train config setting
    config.train.use_gpu = True
    config.train.gpu_index = 0
    config.train.shuffle = True
    config.train.num_workers = 0
    config.train.valid_percent = 0.1
    config.train.lr = 0.001
    config.train.batch_size = 128
    config.train.epoch_size = 2
    config.train.save_model_path = f'/home/ubuntu/likun/image_save_kernels/{config.data.name}_{1}.pt'

    return config


def build_image_caption_config():
    config = BasicConfig()

    # Data config setting
    config.data.name = 'flickr8k'
    config.data.train_caption_path = '/home/ubuntu/likun/image_data/caption/dataset_flickr8k.json'
    config.data.valid_caption_path = None
    config.data.vocab_path = '/home/ubuntu/likun/image_data/vocab/flickr8k_vocab.json'
    config.data.image_root_path = '/home/ubuntu/likun/image_data/flickr8k-images'
    config.data.train_data_nums = "all"
    config.data.valid_data_nums = "all"

    config.data.caption_vocab_size = None

    # Model config setting
    config.model.image_embedding_dim = 224
    config.model.text_embedding_dim = 224
    config.model.text_rnn_hidden_size = 224
    config.model.text_rnn_num_layers = 1
    config.model.text_rnn_is_bi = False
    config.model.rnn_dropout_rate = 0.5

    # IMAGINET model config
    config.model.IMAGINET_alpha = 0.5

    # Train config setting
    config.train.use_gpu = True
    config.train.gpu_index = 0
    config.train.shuffle = True
    config.train.num_workers = 0
    config.train.valid_percent = 0.1
    config.train.lr = 0.001
    config.train.batch_size = 128
    config.train.epoch_size = 2
    config.train.save_model_path = f'/home/ubuntu/likun/image_save_kernels/{config.data.name}_1106.pt'

    return config


def build_image_classify_config():
    config = BasicConfig()

    # Data config setting
    config.data.name = 'clevr'
    config.data.train_question_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_train_questions.json'
    config.data.valid_question_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_val_questions.json'
    config.data.question_vocab_path = '/home/ubuntu/likun/image_data/vocab/clevr_question_vocab.json'
    config.data.answer_vocab_path = '/home/ubuntu/likun/image_data/vocab/clevr_answer_vocab.json'
    config.data.train_image_root_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/images/train'
    config.data.valid_image_root_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/images/val'
    config.data.train_data_nums = 'all'
    config.data.valid_data_nums = 'all'
    config.data.shuffle = True

    config.data.question_vocab_size = None
    config.data.answer_vocab_size = None

    # Model config setting
    config.model.image_embedding_dim = 224
    config.model.question_embedding_dim = 224
    config.model.question_rnn_hidden_size = 256
    config.model.question_rnn_num_layers = 1
    config.model.question_rnn_is_bi = False
    config.model.rnn_dropout_rate = 0.5

    # Train config setting
    config.train.use_gpu = True
    config.train.gpu_index = 0
    config.train.shuffle = True
    config.train.num_workers = 0
    config.train.valid_percent = 0.1
    config.train.lr = 0.001
    config.train.batch_size = 128
    config.train.epoch_size = 2
    config.train.save_model_path = f'/home/ubuntu/likun/image_save_kernels/{config.data.name}_1106.pt'

    return config


def build_text_classify_config():
    config = BasicConfig()

    # Data config setting
    config.data.name = 'dbpedia'
    config.data.train_text_path = '/home/ubuntu/likun/nlp_data/text_classify/dbpedia_csv/train.csv'
    config.data.valid_text_path = '/home/ubuntu/likun/nlp_data/text_classify/dbpedia_csv/train.csv'
    config.data.test_text_path = '/home/ubuntu/likun/nlp_data/text_classify/dbpedia_csv/train.csv'

    # config.data.vocab_path = '/home/ubuntu/likun/vocab/dbpedia_vocab.json'
    config.data.vocab_path = '/home/ubuntu/likun/nlp_pretrained/bert-google-uncase-base/vocab.txt'
    config.data.label_vocab_path = '/home/ubuntu/likun/vocab/dbpedia_label_vocab.json'

    config.data.vocab_size = None
    config.data.num_labels = None

    config.data.train_data_nums = 10000
    config.data.valid_data_nums = 10000
    config.data.test_data_nums = 1000
    config.data.train_use_labels = 'all'
    config.data.valid_use_labels = 'all'
    config.data.test_use_labels = 'all'
    config.data.train_single_label_max_data_nums = 'all'
    config.data.valid_single_label_max_data_nums = 'all'
    config.data.test_single_label_max_data_nums = 'all'

    # Model config setting
    config.model.pre_train_model_path = '/home/ubuntu/likun/nlp_pretrained/bert-google-uncase-base'
    config.model.text_embedding_dim = 224
    config.model.text_rnn_hidden_size = 224
    config.model.text_rnn_num_layers = 1
    config.model.text_rnn_is_bi = False
    config.model.rnn_dropout_rate = 0.5

    config.model.text_cnn_filter_sizes = (1, 2, 3, 4)
    config.model.text_cnn_num_filters = 3
    config.model.text_cnn_pooling_method = 'max'

    # Train config setting
    config.train.use_gpu = True
    config.train.gpu_index = 1
    config.train.shuffle = True
    config.train.num_workers = 0

    config.train.valid_percent = 0.1
    config.train.lr = 0.001
    config.train.batch_size = 128
    config.train.epoch_size = 5
    config.train.save_model = False
    config.train.save_model_path = f'/home/ubuntu/likun/image_save_kernels/{config.data.name}_1106.pt'

    # Predict config setting
    config.predict.use_gpu = True
    config.predict.gpu_index = 1
    config.predict.num_workers = 0
    config.predict.model_path = '/home/ubuntu/likun/nlp_pretrained/bart-large-mnli'
    return config