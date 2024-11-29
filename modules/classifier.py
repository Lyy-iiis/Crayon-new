import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, feature_shape, num_pred_heads):
        super(Classifier, self).__init__()
        if len(feature_shape) > 1:
            raise NotImplementedError()
        self.fc = nn.Linear(feature_shape[0], num_pred_heads)

    def forward(self, x):
        return self.fc(x)
