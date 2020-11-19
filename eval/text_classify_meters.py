from utils import meter
from utils import evaluates
import numpy as np

class DBpediaConceptMeter(meter.ValidMeter):
    def __init__(self):
        super(DBpediaConceptMeter, self).__init__()

    def update(self, loss, feed_dict, output_dict):
        self.loss += loss
        self.true.extend(feed_dict['labels'])
        self.predict.extend(np.argmax(output_dict['predict_logits'].detach().cpu().numpy(), axis=1).tolist())

    def evaluate(self):
        report = evaluates.classification_report(self.true, self.predict)
        print("Loss: {:.2f}".format(self.loss / len(self.true)))
        print(report)