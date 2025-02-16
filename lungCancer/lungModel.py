import torch.nn as nn

class LungModel(nn.Module):
    def __init__(self, features_in, h1=512, h2=256, h3=128, output=1):
        super().__init__()

        # Reduce complexity and add skip connections
        self.input_norm = nn.BatchNorm1d(features_in)

        self.layer1 = nn.Sequential(
            nn.Linear(features_in, h1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(h1),
            nn.Dropout(0.3)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(h2),
            nn.Dropout(0.2)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(h2, h3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(h3),
            nn.Dropout(0.1)
        )

        self.output = nn.Linear(h3, output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_norm(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        out = self.output(x3)
        return self.sigmoid(out)
