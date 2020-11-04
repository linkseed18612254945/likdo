from torch import nn
from models import encoder, decoder

class ImageCaption(nn.Module):
    def __init__(self, config):
        super(ImageCaption, self).__init__()
        self.encoder = encoder.ImageCNNEncoder(config)
        self.decoder = decoder.TextRNNDecoder(config)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, images, captions, lengths):
        monitors = {}
        image_features = self.encoder.forward(images)
        outputs, _ = self.decoder.forward(image_features, captions, lengths)
        target = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]
        loss = self.loss(outputs, target)
        output_dict = {
            "decoder_outputs": outputs
        }
        return loss, monitors, output_dict


