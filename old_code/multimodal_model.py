import torch
from torch import nn
from old_code.model import ImageCNNEncoder, TextRNNEncoder

class BasicAlignModel(nn.Module):
    def __init__(self, image_encoding_size, vocab_size, embedding_size, hidden_size, num_layers):
        super(BasicAlignModel, self).__init__()
        self.image_encoder = ImageCNNEncoder(image_encoding_size)
        self.text_encoder = TextRNNEncoder(vocab_size, embedding_size, hidden_size,
                                           num_layers, output_size=image_encoding_size)

    def forward(self, image, text, lengths):
        _, image_feature = self.image_encoder.forward(image)
        text_feature = self.text_encoder.forward(text, lengths)
        return image_feature, text_feature

