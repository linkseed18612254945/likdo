from torch import nn
from models.common import encoder, decoder


class BasicImageCaption(nn.Module):
    def __init__(self, config):
        super(BasicImageCaption, self).__init__()
        self.encoder = encoder.ImageCNNEncoder(config)
        self.decoder = decoder.TextRNNDecoder(config)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feed_dict):
        images = feed_dict['images']
        captions = feed_dict['captions']
        lengths = feed_dict['lengths']
        monitors = {}
        image_features = self.encoder.forward(images)
        scores, outputs, _ = self.decoder.forward(image_features, captions, lengths)
        target = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]
        loss = self.loss(scores, target)
        output_dict = {
            "decoder_outputs": outputs
        }
        return loss, output_dict, monitors

