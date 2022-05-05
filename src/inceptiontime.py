from inception_block import InceptionBlock
import pytorch_lightning as pl
from torchmetrics import Accuracy


class InceptionTime(pl.LightningModule):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(), use_residual=True):
        super(InceptionTime, self).__init__()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        