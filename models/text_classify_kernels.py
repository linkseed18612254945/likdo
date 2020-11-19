from transformers import BertForSequenceClassification
from torch import nn
import torch

class BertBaseline(nn.Module):
    def __init__(self, config):
        super(BertBaseline, self).__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained(config.model.bert_base_path,
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
