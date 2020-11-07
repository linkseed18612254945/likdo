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

