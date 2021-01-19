import torch as T


class M_ABS(object):
    @T.no_grad()
    def __call__(self, p, grad, lr_in=1, lr_out=1):
        p.addcmul_(p.abs(), grad.mul(lr_in).tanh(), value=-lr_out)

    def __repr__(self):
        return "ABS"


class M_SPOW(object):
    @T.no_grad()
    def __call__(self, p, grad, lr_in=1, lr_out=1):
        p.mul_(T.pow(2, grad.mul(lr_in).tanh().mul(lr_out).mul(-p.sign())))

    def __repr__(self):
        return "SPOW"


class NormalClip(object):
    @T.no_grad()
    def __call__(self, p, grad, lr):
        if p.grad is not None:
            p.add_(grad, alpha=-lr).clamp_min_(0)

    def __repr__(self):
        return "N_CLIP"


class NormalABS(object):
    @T.no_grad()
    def __call__(self, p, grad, lr):
        if p.grad is not None:
            p.add_(grad, alpha=-lr).abs_()

    def __repr__(self):
        return "N_ABS"
