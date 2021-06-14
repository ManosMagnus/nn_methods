import torch as T
from torch.optim.optimizer import Optimizer


class HRMSprop(Optimizer):
    r"""Implements RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """
    def __init__(self,
                 params,
                 u_func,
                 lr_in=1.0,
                 lr_out=1e-2,
                 lr=1e-2,
                 g=1.0,
                 alpha=0.99,
                 eps=1e-8,
                 weight_decay=0,
                 momentum=0,
                 centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_in:
            raise ValueError("Invalid inner learning rate: {}".format(lr_in))
        if not 0.0 <= lr_out:
            raise ValueError("Invalid outer learning rate: {}".format(lr_out))
        if g < 0.0 or g > 1.0:
            raise ValueError("Invalid g value: {}".format(g))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(u_func=u_func,
                        lr_in=lr_in,
                        lr_out=lr_out,
                        lr=lr,
                        g=g,
                        momentum=momentum,
                        alpha=alpha,
                        eps=eps,
                        centered=centered,
                        weight_decay=weight_decay)
        super(HRMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HRMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @T.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with T.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = T.zeros_like(
                        p, memory_format=T.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = T.zeros_like(
                            p, memory_format=T.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = T.zeros_like(
                            p, memory_format=T.preserve_format)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg,
                                             value=-1).sqrt_().add_(
                                                 group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    group['u_func'](p,
                                    buf,
                                    lr_in=group['lr_in'],
                                    lr_out=group['lr_out'],
                                    lr=group['lr'],
                                    g=group['group'])
                    # -> p.add_(buf, alpha=-group['lr'])

                else:
                    group['u_func'](p,
                                    T.div(grad, avg),
                                    lr_in=group['lr_in'],
                                    lr_out=group['lr_out'],
                                    lr=group['lr'],
                                    g=group['g'])
                    # -> p.addcdiv_(grad, avg, value=-group['lr'])

        return loss
