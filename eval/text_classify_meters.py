from utils import meter
from utils import evaluates
import numpy as np

class BasicTextClassifyMeter(meter.ValidMeter):
    def __init__(self, id2label=None):
        super(BasicTextClassifyMeter, self).__init__()
        self.id2label = id2label

    def update(self, loss, feed_dict, output_dict):
        self.loss.update(loss, len(feed_dict))
        self.true.extend([int(label.detach().cpu().numpy()) for label in feed_dict['labels']])
        self.predict.extend(np.argmax(output_dict['predict_logits'].detach().cpu().numpy(), axis=1).tolist())

    def evaluate(self):
        if self.id2label is None:
            report = evaluates.classification_report(self.true, self.predict)
        else:
            report = evaluates.classification_report(self.true, self.predict,
                                                     labels=list(range(len(self.id2label))),
                                                     target_names=self.id2label)
        print("Loss: {:.2f}".format(self.loss.avg))
        print(report)