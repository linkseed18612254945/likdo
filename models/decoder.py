from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['TextRNNDecoder']

class TextRNNDecoder(nn.Module):
    def __init__(self, config):
        super(TextRNNDecoder, self).__init__()
        self.embedding = nn.Embedding(config.data.vocab_size, config.model.text_embedding_dim)
        self.lstm = nn.LSTM(config.model.text_embedding_dim, config.model.text_rnn_hidden_size,
                            config.model.text_rnn_num_layers, batch_first=True)
        self.fc = nn.Linear(config.model.text_rnn_hidden_size, config.data.vocab_size)

    def forward(self, encoder_feature, seqs, lengths):
        seqs_embedding = self.embedding(seqs)
        feature = torch.cat((encoder_feature.unsqueeze(1), seqs_embedding), dim=1)  # (batch_size, img_feature + embedding)
        packed = pack_padded_sequence(feature, lengths, batch_first=True)
        outputs, hiddens = self.lstm(packed)
        outputs, (h, _) = pad_packed_sequence(outputs, batch_first=True)
        outputs = self.fc(outputs.data)
        return outputs, h

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