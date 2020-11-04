import torch
from torch import nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class VQANoImageModule(nn.Module):
    def __init__(self, q_vocab_size, a_vocab_size, embedding_size, hidden_size, num_layers, output_size):
        super(VQANoImageModule, self).__init__()
        self.encoder = LSTMEncoder(q_vocab_size, embedding_size, hidden_size, num_layers, output_size)
        self.fc = nn.Linear(output_size, a_vocab_size)

    def forward(self, image, question, lengths):
        question_feature = self.encoder(question, lengths)
        output = self.fc(question_feature)
        return output

class ResnetImageEncoder(nn.Module):
    def __init__(self, image_feature_size=None):
        super(ResnetImageEncoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        image_feature_size = resnet.fc.in_features if image_feature_size is None else image_feature_size
        self.fc = nn.Linear(resnet.fc.in_features, image_feature_size)
        self.bn = nn.BatchNorm1d(image_feature_size, momentum=0.01)

    def forward(self, img):
        with torch.no_grad():
            feature = self.resnet(img)
        feature = feature.reshape(feature.size(0), -1)
        feature = self.fc(feature)
        feature = self.bn(feature)
        return feature

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, output_size, dropout=0.1, bi=False):
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=bi)
        self.output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, question, lengths):
        feature = self.embedding(question)
        feature = self.dropout(feature)
        packed = pack_padded_sequence(feature, lengths, batch_first=True)
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        output = self.output(outputs[:, -1, :].squeeze())
        return output