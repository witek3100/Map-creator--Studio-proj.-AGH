from torch.nn import Module, Linear, BatchNorm1d, Sigmoid, Dropout
from torch.hub import load


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

        base = load('pytorch/vision:v0.10.0', 'resnet34')
        #         conv_stem_weights = base.state_dict()['conv_stem.weight'].sum(dim=1, keepdim=True)
        #         base.conv_stem = nn.Conv2d(1, 72, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #         base.state_dict()['conv_stem.weight'] = conv_stem_weights

        self.base = base
        self.hidden1 = Linear(1000, 512)
        self.bn1 = BatchNorm1d(512)
        self.hidden2 = Linear(512, 256)
        self.bn2 = BatchNorm1d(256)
        self.hidden3 = Linear(256, 128)
        self.bn3 = BatchNorm1d(128)
        self.out = Linear(128, 1)
        self.sig = Sigmoid()

        self.dropout = Dropout(0.5)

    def forward(self, x):
        x = self.base(x)
        x = self.hidden1(x)
        x = self.dropout(x)
        x = self.bn1(x)
        x = self.hidden2(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.hidden3(x)
        x = self.dropout(x)
        x = self.bn3(x)
        x = self.out(x)
        out = self.sig(x)

        return out
