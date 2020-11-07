from torch import nn
from models.common import encoder, decoder


class OnlyText(nn.Module):
    def __init__(self, config):
        super(OnlyText, self).__init__()
        self.encoder = encoder.TextRNNEncoder(config)