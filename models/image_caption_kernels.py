from torch import nn
from models.common import encoder, decoder


class BasicImageCaption(nn.Module):
    def __init__(self, config):
        super(BasicImageCaption, self).__init__()
        self.encoder = encoder.ImageCNNEncoder(image_embedding_dim=config.model.image_embedding_dim)
        self.decoder = decoder.TextRNNDecoder(vocab_size=config.data.caption_vocab_size,
                                              embedding_dim=config.model.text_embedding_dim,
                                              hidden_size=config.model.text_rnn_hidden_size,
                                              num_layers=config.model.text_rnn_num_layers,
                                              is_bi=config.model.text_rnn_is_bi)
        self.fc = nn.Linear(config.model.text_rnn_hidden_size * 2 if config.model.text_rnn_is_bi
                            else config.model.text_rnn_hidden_size,
                            config.data.caption_vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feed_dict):
        images = feed_dict['images']
        captions = feed_dict['captions']
        lengths = feed_dict['lengths']
        monitors = {}
        image_features = self.encoder.forward(images)
        output, outputs, _ = self.decoder.forward(image_features, captions, lengths)
        scores = self.fc(output)
        target = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]
        loss = self.loss(scores, target)
        output_dict = {
            "decoder_outputs": outputs
        }
        return loss, output_dict, monitors


class IMAGINET(nn.Module):
    def __init__(self, config):
        super(IMAGINET, self).__init__()
        self.image_encoder = encoder.ImageCNNEncoder(image_embedding_dim=config.model.image_embedding_dim)
        self.caption_decoder = decoder.TextRNNDecoder(vocab_size=config.data.caption_vocab_size,
                                                      embedding_dim=config.model.text_embedding_dim,
                                                      hidden_size=config.model.text_rnn_hidden_size,
                                                      num_layers=config.model.text_rnn_num_layers,
                                                      is_bi=config.model.text_rnn_is_bi)
        self.alpha = config.model.IMAGINET_alpha
        self.fc = nn.Linear(config.model.text_rnn_hidden_size * 2 if config.model.text_rnn_is_bi
                            else config.model.text_rnn_hidden_size,
                            config.data.caption_vocab_size)
        self.caption_predict_loss = nn.CrossEntropyLoss()
        self.image_caption_similar_loss = nn.MSELoss()

    def forward(self, feed_dict):
        monitors = {}
        image_feature = self.image_encoder.forward(feed_dict['images'])
        _, outputs, last_hidden = self.caption_decoder.forward(encoder_feature=None,
                                                               seqs=feed_dict['captions'],
                                                               lengths=feed_dict['lengths'])
        last_hidden = last_hidden.squeeze()
        predict_use_length = [lenth - 1 for lenth in feed_dict['lengths']]
        pack_outputs = nn.utils.rnn.pack_padded_sequence(outputs, predict_use_length, batch_first=True)[0]
        pack_outputs = self.fc(pack_outputs)
        targets = nn.utils.rnn.pack_padded_sequence(feed_dict['captions'][:, 1:], predict_use_length, batch_first=True)[0]
        loss_1 = self.caption_predict_loss(pack_outputs, targets)
        loss_2 = self.image_caption_similar_loss(last_hidden, image_feature)
        loss = loss_1 * self.alpha + loss_2 + (1 - self.alpha)
        output_dict = {
            "caption_outputs": outputs
        }
        return loss, output_dict, monitors
