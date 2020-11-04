import torch
from torch import nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ImageCaptionModule(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, vocab_size):
        super(ImageCaptionModule, self).__init__()
        self.encoder = ImageCNNEncoder(embedding_size)
        self.decoder = DecoderRNN(embedding_size, hidden_size, num_layers, vocab_size)

    def forward(self, img, caption, lengths):
        img_feature = self.encoder.forward(img)
        output = self.decoder.forward(img_feature, caption, lengths)
        return output

    def sample(self, img, max_length, eos_index=None):
        img_feature = self.encoder.forward(img)
        sampled_ids = self.decoder.sample(img_feature, max_length, eos_index)
        return sampled_ids

class ImageCNNEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(ImageCNNEncoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size, momentum=0.01)

    def forward(self, img):
        with torch.no_grad():
            feature = self.resnet(img)
        feature = feature.reshape(feature.size(0), -1)
        feature = self.fc(feature)
        feature = self.bn(feature)
        return feature

class TextRNNEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, output_size=None, dropout=0.1, bi=False):
        super(TextRNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=bi)
        if output_size is not None:
            self.linear = nn.Linear(hidden_size * 2 if bi else hidden_size, output_size)
        else:
            self.linear = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, question, lengths):
        feature = self.embedding(question)
        feature = self.dropout(feature)
        packed = pack_padded_sequence(feature, lengths, batch_first=True)
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        if self.linear is not None:
            last_feature = self.linear(outputs[:, -1, :].squeeze())
        else:
            last_feature = outputs[:, -1, :].squeeze()
        return outputs, last_feature


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, img_feature, caption, lengths):
        caption = self.embedding(caption)
        feature = torch.cat((img_feature.unsqueeze(1), caption), dim=1)  # (batch_size, img_feature + embedding)
        packed = pack_padded_sequence(feature, lengths, batch_first=True)
        output, hidden = self.lstm(packed)
        output = self.fc(output.data)
        return output

    def sample(self, img_feature, max_length, eos_index=None):
        inputs = img_feature.unsqueeze(1)
        hidden = None
        sampled_ids = []
        for i in range(max_length):
            output, hidden = self.lstm(inputs, hidden)
            output = self.fc(output.squeeze(1))
            _, predicted = output.max(1)
            sampled_ids.append(predicted)
            if eos_index is not None and eos_index == predicted:
                break
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

class CNNRawEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(CNNRawEncoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((embedding_size, embedding_size))

    def forward(self, image):
        image = self.resnet(image)
        image_feature = self.pool(image)
        # image_feature = image_feature.permute(0, 2, 3, 1)
        return image_feature

class RNNAttentionDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, image_feature_size):
        super(RNNAttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm_step = nn.LSTMCell(embedding_size + image_feature_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self):
        pass

