import torch as T


class NNLinear(T.nn.Module):
    def __init__(
        self,
        fan_in,
        fan_out,
        w_pos,
        w_neg,
        bias,
        alpha,
        activation_shift,
    ):
        super().__init__()
        super().cuda()

        self.l_pos = T.nn.Linear(fan_in, fan_out, bias=False)
        self.l_neg = T.nn.Linear(fan_in, fan_out, bias=False)

        self.l_pos.weight = T.nn.Parameter(w_pos.cuda())
        self.l_neg.weight = T.nn.Parameter(w_neg.cuda())

        self.bias = T.nn.Parameter(bias.cuda())

        # self.weight = T.nn.Parameter(self.l_pos.weight + self.l_neg.weight,
        #                              True)

        self.register_buffer('alpha', alpha)

        self.register_buffer('activation_shift', activation_shift)

    def forward(self, x):
        x_t = self.alpha - x
        x = self.l_pos(x) + self.l_neg(x_t) + self.bias
        x = x - self.activation_shift
        return x

    def get_weight(self):
        return self.l_pos.weight + self.l_neg.weight
