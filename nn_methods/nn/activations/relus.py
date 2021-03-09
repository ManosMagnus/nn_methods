import torch as T


class ReLUN(T.nn.Module):
    def __init__(self, bound=6):
        super().__init__()
        self.bound = bound

    def forward(self, x):
        return T.clamp_max(T.relu(x), self.bound)

    def __repr__(self):
        return "ReLU({})".format(self.bound)


class ReLU6(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.bound = 6

    def forward(self, x):
        return T.clamp_max(T.relu(x), self.bound)

    def __repr__(self):
        return "ReLU6"


class ReLU2(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.bound = 2

    def forward(self, x):
        return T.clamp_max(T.relu(x), self.bound)

    def __repr__(self):
        return "ReLU6"
