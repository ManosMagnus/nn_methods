import torch as T
from nn_methods.nn.nn_conv2d import NNConv2d
from nn_methods.nn.nn_layer import NNLinear


def nn_transformation(model):
    b_new_list = [None for _ in range(len(model.layers))]
    act_shift_list = [None for _ in range(len(model.layers))]

    nn_model = model.get_nn_net()
    for i in range(len(model.layers)):
        cur_layer = model.layers[i]
        classname = cur_layer.__class__.__name__

        if classname.find('Linear') != -1:
            sum_dim = 1
        elif classname.find('Conv2d') != -1:
            sum_dim = (1, 2, 3)

        w_neg = T.clamp_max(cur_layer.weight, 0)
        b_tilde = cur_layer.bias - model.alpha[i] * T.sum(T.abs(w_neg),
                                                          dim=sum_dim)
        b_new, act_shift = calc_b_new(cur_layer, b_tilde)

        b_new_list[i] = b_new
        act_shift_list[i] = act_shift

        w_neg_abs = T.abs(w_neg)
        w_pos = T.clamp_min(cur_layer.weight, 0)

        if classname.find('Linear') != -1:
            new_layer = NNLinear(
                cur_layer.in_features,
                cur_layer.out_features,
                w_pos,
                w_neg_abs,
                b_new,
                model.alpha[i],
                act_shift,
            )
        elif classname.find('Conv2d') != -1:
            new_layer = NNConv2d(
                cur_layer.in_channels,
                cur_layer.out_channels,
                cur_layer.kernel_size,
                w_pos,
                w_neg_abs,
                b_new,
                model.alpha[i],
                act_shift,
                stride=cur_layer.stride,
                padding=cur_layer.padding,
                dilation=cur_layer.dilation,
                groups=cur_layer.groups,
                padding_mode=cur_layer.padding_mode,
            )
        else:
            raise Exception("This layer type cannot be converted")
        nn_model.add_layer(new_layer)
    return nn_model


def calc_b_new(layer, b_tilde):
    with T.no_grad():
        act_shift = T.max(T.abs(b_tilde))

        # For b_tilde < 0 replace biases with b_tilde
        b_new = T.scatter(layer.bias, 0,
                          T.where(b_tilde < 0)[0], b_tilde[b_tilde < 0])

        # For b_tilde < 0 update biases
        b_new = T.scatter_add(-T.abs(b_new), 0,
                              T.where(b_tilde < 0)[0],
                              act_shift.expand_as(T.where(b_tilde < 0)[0]))
        # For b_tilde > 0 update biases
        b_new = T.scatter_add(b_new, 0,
                              T.where(b_tilde > 0)[0],
                              act_shift.expand_as(T.where(b_tilde > 0)[0]))

    return b_new, act_shift
