import torch
import torch.nn as nn
import torch.nn.functional as F
from .morph_layers2D_torch import *
import hypergraph
import copy
import contextlib
from torch.distributed import ReduceOp
from torch.nn.modules.batchnorm import _BatchNorm

# from torch_geometric.data import Data
# from torch_geometric.nn import SAGEConv



class Discriminator(nn.Module):

    def __init__(self, inchannel, outchannel, num_classes, patch_size,pad=False):
        super(Discriminator, self).__init__()
        dim = 512
        self.patch_size = patch_size
        # self.lambda1 = torch.nn.Parameter(torch.FloatTensor([0.95]), requires_grad=True)
        self.inchannel = inchannel
        self.matching_cfg = 'o2o'
        # self.node_affinity = Affinity(dim)
        self.matching_loss = nn.MSELoss(reduction='sum')
        self.with_hyper_graph = True
        self.num_hyper_edge = 3
        self.angle_eps = 1e-3
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        if pad:
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())
        self.hgnn = hypergraph.HyperGraph(emb_dim=dim, K_neigs=self.num_hyper_edge)
        self.gnn = hypergraph.MultiHeadAttention(model_dim=dim,num_heads=1)


    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1


    def forward(self, x, mode='test'):

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.contiguous().view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        # node_s, edge_s = self.gnn(out3, out3, out3)
        # out_fus = self.lambda1*out3+(1-self.lambda1)*node_s
        # out4 = self.relu4(self.fc2(0.95*out3+0.05*node_s))
        out4 = self.relu4(self.fc2(out3))
        # out4 = self.relu4(self.fc2(out_fus))

        # matching_loss_affinity, affinity = self._forward_aff(node_s, node_t, src_label, tar_label)
        # matching_loss_quadratic = self._forward_qu(node_s, node_t, edge_s.detach(), edge_t.detach(), affinity)
        # edge_s = 0
        # proj = F.normalize(self.pro_head(out4))
        # return proj

        if mode == 'test':
            # clss = self.cls_head_src(out4)
            proj = self.cls_head_src(out4)
            return proj
        elif mode == 'train':
            proj = F.normalize(self.pro_head(out4))
            # proj = F.normalize(node_s)
            clss = self.cls_head_src(out4)

            return clss, proj



class MorphNet(nn.Module):
    def __init__(self, inchannel):
        super(MorphNet, self).__init__()
        num = 1
        kernel_size = 3
        self.conv1 = nn.Conv2d(inchannel, num, kernel_size=1, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.Erosion2d_1 = Erosion2d(num, num, kernel_size, soft_max=False)
        self.Dilation2d_1 = Dilation2d(num, num, kernel_size, soft_max=False)
        self.Erosion2d_2 = Erosion2d(num, num, kernel_size, soft_max=False)
        self.Dilation2d_2 = Dilation2d(num, num, kernel_size, soft_max=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        xop_2 = self.Dilation2d_1(self.Erosion2d_1(x))
        xcl_2 = self.Erosion2d_2(self.Dilation2d_2(x))
        x_top = x - xop_2
        x_blk = xcl_2 - x
        x_morph = torch.cat((x_top, x_blk, xop_2, xcl_2), 1)

        return x_morph



def get_optimizer(name, params, **kwargs):
    name = name.lower()
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    optim_cls = optimizers[name]

    return optim_cls(params, **kwargs)


class SchedulerBase:
    def __init__(self, T_max, max_value, min_value=0.0, init_value=0.0, warmup_steps=0, optimizer=None):
        super(SchedulerBase, self).__init__()
        self.t = 0
        self.min_value = min_value
        self.max_value = max_value
        self.init_value = init_value
        self.warmup_steps = warmup_steps
        self.total_steps = T_max

        # record current value in self._last_lr to match API from torch.optim.lr_scheduler
        self._last_lr = [init_value]

        # If optimizer is not None, will set learning rate to all trainable parameters in optimizer.
        # If optimizer is None, only output the value of lr.
        self.optimizer = optimizer

    def step(self):
        if self.t < self.warmup_steps:
            value = self.init_value + (self.max_value - self.init_value) * self.t / self.warmup_steps
        elif self.t == self.warmup_steps:
            value = self.max_value
        else:
            value = self.step_func()
        self.t += 1

        # value = self.max_value

        # apply the lr to optimizer if it's provided
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value

        self._last_lr = [value]

        # print(value)

        return value

    def step_func(self):
        pass

    def lr(self):
        return self._last_lr[0]

class LinearScheduler(SchedulerBase):
    def step_func(self):
        value = self.max_value + (self.min_value - self.max_value) * (self.t - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps)
        return value


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

class SAGM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, alpha, rho_scheduler, adaptive=False, perturb_eps=1e-12,
                 grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAGM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.alpha = alpha

        # initialize self.rho_t
        self.update_rho_t()

        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = (rho / (grad_norm + self.perturb_eps) - self.alpha)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                sam_grad = self.state[p]['old_g'] * 0.5 - p.grad * 0.5
                p.grad.data.add_(sam_grad)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:

            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )

        else:

            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )

        return norm

    # def norm(tensor_list: List[torch.tensor], p=2):
    #     """Compute p-norm for tensor list"""
    #     return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value



class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self,  num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        # self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError



    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone

class SAGM_DG(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    # def __init__(self, input_shape, num_classes, num_domains, hparams):
    #     assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
    #     super().__init__(input_shape, num_classes, num_domains, hparams)
    def __init__(self,num_classes, num_domains, hparams):
        super().__init__(num_classes, num_domains, hparams)
        # self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.featurizer = Discriminator(num_classes=num_classes,inchannel=hparams['n_bands'],outchannel=hparams['pro_dim'],patch_size=hparams['patch_size'])
        self.classifier = nn.Linear(hparams['pro_dim'], num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.lr_scheduler = LinearScheduler(T_max=5000, max_value=self.hparams["lr"],
                                            min_value=self.hparams["lr"], optimizer=self.optimizer)

        self.rho_scheduler = LinearScheduler(T_max=5000, max_value=0.05,
                                             min_value=0.05)

        self.SAGM_optimizer = SAGM(params=self.network.parameters(), base_optimizer=self.optimizer,
                                   model=self.network,
                                   alpha=self.hparams["alpha"], rho_scheduler=self.rho_scheduler, adaptive=False)

    def update(self, x, y, **kwargs):
        all_x = x
        all_y = y

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        self.SAGM_optimizer.set_closure(loss_fn, all_x, all_y)
        predictions, loss = self.SAGM_optimizer.step()
        self.lr_scheduler.step()
        self.SAGM_optimizer.update_rho_t()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)