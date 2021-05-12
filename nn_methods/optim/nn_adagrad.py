import math

import torch
from torch import Tensor
from torch.optim.functional import _make_sparse
from torch.optim.functional.typing import List
from torch.optim.optimizer import Optimizer


class Adagrad(Optimizer):
    """Implements Adagrad algorithm.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """
    def __init__(self,
                 params,
                 lr=1e-2,
                 lr_decay=0,
                 weight_decay=0,
                 initial_accumulator_value=0,
                 eps=1e-10):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr,
                        lr_decay=lr_decay,
                        eps=eps,
                        weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(Adagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(
                    p,
                    initial_accumulator_value,
                    memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_sums = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    state_sums.append(state['sum'])
                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            adagrad(params_with_grad, grads, state_sums, state_steps,
                    group['lr'], group['weight_decay'], group['lr_decay'],
                    group['eps'])

        return loss


def adagrad(
    u_func,
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[int],
    lr_in: float,
    lr_out: float,
    g: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
):
    r"""Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    for (param, grad, state_sum, step) in zip(params, grads, state_sums,
                                              state_steps):
        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError(
                    "weight_decay option is not compatible with sparse gradients"
                )
            grad = grad.add(param, alpha=weight_decay)

        clr = lr_out / (1 + (step - 1) * lr_decay)

        if grad.is_sparse:
            grad = grad.coalesce(
            )  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

            state_sum.add_(_make_sparse(grad, grad_indices,
                                        grad_values.pow(2)))
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt_().add_(eps)
            sparse_grad = _make_sparse(grad, grad_indices,
                                       grad_values / std_values)
            param.add_(sparse_grad, lr_out=lr_out, lr_in=lr_in, g=g)

            # -> param.add_(_make_sparse(grad, grad_indices,
            #                         grad_values / std_values),
            #                         alpha=-clr)
        else:
            state_sum.addcmul_(grad, grad, value=1)
            std = state_sum.sqrt().add_(eps)
            u_func(param,
                   torch.div(grad, std),
                   lr_in=lr_in,
                   lr_out=lr_out,
                   g=g)

            # -> param.addcdiv_(grad, std, value=-clr)
