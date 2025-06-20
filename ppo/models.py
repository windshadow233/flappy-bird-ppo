from torch import nn


class ActionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(3136, 512),
            nn.Tanh(),
            nn.Linear(512, 2)
        )

    def forward(self, state):
        x = self.conv(state)
        x = self.linear(x)
        return x.log_softmax(dim=-1)


class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, state):
        x = self.conv(state)
        x = self.linear(x)
        return x
