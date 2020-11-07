

import torch
from torch import nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['ImageCNNEncoder', 'TextRNNEncoder']

class ImageCNNEncoder(nn.Module):
    def __init__(self, config):
        super(ImageCNNEncoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, config.model.image_embedding_dim)
        self.bn = nn.BatchNorm1d(config.model.image_embedding_dim, momentum=0.01)

    def forward(self, img):
        with torch.no_grad():
            feature = self.resnet(img)
        feature = feature.reshape(feature.size(0), -1)
        feature = self.fc(feature)
        feature = self.bn(feature)
        return feature

class TextRNNEncoder(nn.Module):
    def __init__(self, config):
        super(TextRNNEncoder, self).__init__()
        self.embedding = nn.Embedding(config.data.vocab_size, config.model.text_embedding_dim)
        self.rnn = nn.LSTM(config.model.text_embedding_dim, config.model.text_rnn_hidden_size,
                           config.model.text_rnn_num_layers, batch_first=True,
                           bidirectional=config.model.text_rnn_is_bi)
        self.dropout = nn.Dropout(p=config.model.rnn_dropout_rate)

    def forward(self, seqs, lengths):
        feature = self.embedding(seqs)
        feature = self.dropout(feature)
        packed = pack_padded_sequence(feature, lengths, batch_first=True)
        outputs, hidden = self.rnn(packed)
        outputs, (h, _) = pad_packed_sequence(outputs, batch_first=True)
        return outputs, h
