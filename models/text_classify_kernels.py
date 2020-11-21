from transformers import BertForSequenceClassification
from torch import nn
import torch
from models.common import encoder

class BertBaseline(nn.Module):
    def __init__(self, config):
        super(BertBaseline, self).__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained(config.model.pre_train_model_path,
                                                                        num_labels=config.data.num_labels,
                                                                        output_attentions=False, output_hidden_states=False)

    def forward(self, feed_dict):
        texts = feed_dict['texts']
        labels = feed_dict['labels']
        monitors = {}
        attention_mask = (texts > 0)
        loss, logits = self.bert_model.forward(texts, token_type_ids=None, attention_mask=attention_mask, labels=labels)
        output_dict = {
            'predict_logits': logits
        }
        return loss, output_dict, monitors


class RNNBaseline(nn.Module):
    def __init__(self, config):
        super(RNNBaseline, self).__init__()
        self.encoder = encoder.TextRNNEncoder(vocab_size=config.data.vocab_size, embedding_dim=config.model.text_embedding_dim,
                                              hidden_size=config.model.text_rnn_hidden_size, num_layers=config.model.text_rnn_num_layers,
                                              is_bi=config.model.text_rnn_is_bi)
        self.fc = nn.Linear(config.model.text_rnn_hidden_size * 2 if config.model.text_rnn_is_bi else config.model.text_rnn_hidden_size,
                            config.data.num_labels)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feed_dict):
        monitors = {}
        texts = feed_dict['texts']
        labels = feed_dict['labels']
        output, h = self.encoder.forward(texts)
        logits = self.fc(h.squeeze(0))
        loss = self.loss(logits, labels)
        output_dict = {
            "predict_logits": logits
        }
        return loss, output_dict, monitors


