import torch as T
from torch.distributions import Normal
from torch.nn.init import (kaiming_normal, kaiming_normal_, xavier_normal,
                           xavier_normal_)


class NoInit(object):
    def __init__(self):
        pass

    def __call__(self, layer):
        pass

    def __repr__(self):
        return "NoInit"

    def __str__(self):
        return self.__repr__()


class NormalInitModel(object):
    def __init__(self, mean=0.0, std=1.0, bias=0.0):
        self.mean = mean
        self.std = std
        self.bias = bias
        self.normal_init_layer = NormalInitLayer(mean, std, bias)

    def __call__(self, model):
        for layer in model.layers:
            self.normal_init_layer(layer)
            print("Min weight:", layer.weight.min())

    def __repr__(self):
        return "NormalInit({},{}),b={}".format(self.mean, self.std, self.bias)


class NormalInitLayer(object):
    def __init__(self, mean=0.0, std=1.0, bias=0.0):
        self.mean = mean
        self.std = std
        self.bias = bias

    def __call__(self, layer):
        T.nn.init.normal_(layer.weight, mean=self.mean, std=self.std)
        T.nn.init.constant_(layer.bias, self.bias)

    def __repr__(self):
        return "NormalInit({},{}),b={}".format(self.mean, self.std, self.bias)


class NormalInitAbsModel(object):
    def __init__(self, mean=0.0, std=1.0, bias=0.0):
        self.mean = mean
        self.std = std
        self.bias = bias
        self.normal_abs_init_layer = NormalAbsInitLayer(mean, std, bias)

    def __call__(self, model):
        for layer in model.layers:
            self.normal_abs_init_layer(layer)

    def __repr__(self):
        return "NormalInit({},{}),b={}".format(self.mean, self.std, self.bias)


class NormalAbsInitLayer(object):
    def __init__(self, mean=0.0, std=1.0, bias=0.0):
        self.mean = mean
        self.std = std
        self.bias = bias
        self.normal_init = NormalInitLayer(mean, std, bias)

    def __call__(self, layer):
        self.normal_init(layer)
        weights_abs(layer)

    def __repr__(self):
        return "NormalAbsInit({},{}),b={}".format(self.mean, self.std,
                                                  self.bias)


class UniformInitModel(object):
    def __init__(self, a=0.0, b=1.0, bias=0.0):
        self.a = a
        self.b = b
        self.bias = bias
        self.uniform_init_layer = UniformInitLayer(a, b, bias)

    def __call__(self, model):
        for layer in model.layers:
            self.uniform_init_layer(layer)
            print("Min weight:", layer.weight.min())

    def __repr__(self):
        return "NormalInit({},{}),b={}".format(self.a, self.b, self.bias)


class UniformInitLayer(object):
    def __init__(self, a=0.0, b=1.0, bias=0.0):
        self.a = a
        self.b = b
        self.bias = bias

    def __call__(self, layer):
        T.nn.init.uniform_(layer.weight, a=self.a, b=self.b)
        T.nn.init.constant_(layer.bias, self.bias)

    def __repr__(self):
        return "NormalInit({},{}),b={}".format(self.a, self.b, self.bias)


def create_custom_initializer(std):
    return lambda m: weights_init_normal(m, std)


@T.no_grad()
def weights_init_normal(m, std=0.5):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, std)
        m.bias.data.fill_(0)


@T.no_grad()
def weights_xavier_normal(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        xavier_normal_(m.weight)
        m.bias.data.fill_(0)


@T.no_grad()
def weights_kaiming_normal(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


@T.no_grad()
def weights_xavier_abs(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        xavier_normal_(m.weight)
        m.weight.abs_()
        m.bias.data.fill_(0)


@T.no_grad()
def weights_kaiming_abs(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        kaiming_normal_(m.weight)
        m.weight.abs_()
        m.bias.data.fill_(0)


@T.no_grad()
def weights_abs(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.abs_()
        m.bias.abs_()


@T.no_grad()
def weights_clip(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.clamp_min_(0)
        m.bias.clamp_min_(0)
