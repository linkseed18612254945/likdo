from torch import nn
from models.common import encoder, decoder


class BasicParallelMap(nn.Module):
    def __init__(self, config):
        super(BasicParallelMap, self).__init__()
        self.encoder = nn.Sequential(
            encoder.ImageCNNEncoder(image_embedding_dim=config.model.image_embedding_dim),
            nn.Tanh(),
            nn.Linear(config.model.image_embedding_dim, config.model.text_vector_dim)
        )
        self.loss = nn.MSELoss()

    def forward(self, feed_dict):
        monitors = {}

        images = feed_dict['images']
        targets = feed_dict['targets']
        image_features = self.encoder.forward(images)
        loss = self.loss(image_features, targets)
        output_dict = {
            "image_feature": image_features
        }
        return loss, output_dict, monitors