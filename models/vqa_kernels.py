from torch import nn
from models.common import encoder, decoder


class OnlyText(nn.Module):
    def __init__(self, config):
        super(OnlyText, self).__init__()
        self.encoder = encoder.TextRNNEncoder(vocab_size=config.data.question_vocab_size, embedding_dim=config.model.question_embedding_dim,
                                              hidden_size=config.model.question_rnn_hidden_size, num_layers=config.model.question_rnn_num_layers)
        self.fc = nn.Linear(config.model.question_rnn_hidden_size * 2 if config.model.question_rnn_is_bi
                            else config.model.question_rnn_hidden_size, config.data.answer_vocab_size)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, feed_dict):
        monitors = {}
        question_feature = self.encoder.forward(feed_dict['questions'], feed_dict['question_lengths'])
        scores = self.fc(question_feature)
        loss = self.loss_function(scores, feed_dict['answers'])
        output_dict = {
            'question_feature': question_feature
        }
        return loss, output_dict, monitors