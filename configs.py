from utils.container import BasicConfig

def build_image_caption_config():
    config = BasicConfig()

    # Data config setting
    config.data.name = 'flickr8k'
    config.data.train_caption_path = '/home/ubuntu/likun/image_data/caption/dataset_flickr8k.json'
    config.data.valid_caption_path = None
    config.data.vocab_path = '/home/ubuntu/likun/image_data/vocab/flickr8k_vocab.json'
    config.data.image_root_path = '/home/ubuntu/likun/image_data/flickr8k-images'
    config.data.train_data_nums = 1000
    config.data.valid_data_nums = 1000

    config.data.caption_vocab_size = None

    # Model config setting
    config.model.image_embedding_dim = 224
    config.model.text_embedding_dim = 224
    config.model.text_rnn_hidden_size = 256
    config.model.text_rnn_num_layers = 1
    config.model.text_rnn_is_bi = False
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
    config.train.save_model_path = '/home/ubuntu/likun/image_save_kernels/image_caption_f8k_1106.pt'

    return config

def build_vqa_config():
    config = BasicConfig()

    # Data config setting
    config.data.name = 'clevr'
    config.data.train_question_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_train_questions.json'
    config.data.valid_question_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/questions/CLEVR_val_questions.json'
    config.data.question_vocab_path = '/home/ubuntu/likun/image_data/vocab/clevr_question_vocab.json'
    config.data.answer_vocab_path = '/home/ubuntu/likun/image_data/vocab/clevr_answer_vocab.json'
    config.data.train_image_root_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/images/train'
    config.data.valid_image_root_path = '/home/ubuntu/likun/image_data/CLEVR_v1.0/images/val'
    config.data.train_data_nums = 1000
    config.data.valid_data_nums = 1000

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
    config.train.save_model_path = '/home/ubuntu/likun/image_save_kernels/image_caption_f8k_1106.pt'

    return config
