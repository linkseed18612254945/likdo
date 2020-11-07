from torch import nn
from models.common import encoder, decoder


class VQA(nn.Module):
    def __init__(self, config):
        super(VQA, self).__init__()
        self.encoder = encoder.TextRNNEncoder()