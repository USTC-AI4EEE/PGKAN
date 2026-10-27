import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from utils.util import AverageMeter, get_logger, eval_metrix
import os
from .efficient_KAN import KAN

# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'  # Now using self.device

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class MLP(nn.Module):
    def __init__(self, input_dim=17, output_dim=1, layers_num=4, hidden_dim=50, droupout=0.2):
        super(MLP, self).__init__()

        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                self.layers.append(Sin())
            elif i == layers_num - 1:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(Sin())
                self.layers.append(nn.Dropout(p=droupout))
        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        x = self.net(x)
        return x

class Predictor(nn.Module):
    def __init__(self, input_dim=40):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, 32),
            Sin(),
            nn.Linear(32, 1)
        )
        self.input_dim = input_dim

    def forward(self, x):
        return self.net(x)

class Solution_u(nn.Module):
    def __init__(self):
        super(Solution_u, self).__init__()
        # self.encoder = MLP(input_dim=17, output_dim=32, layers_num=3, hidden_dim=60, droupout=0.2)
        # self.predictor = Predictor(input_dim=32)
        self.encoder = KAN(layers_hidden=[17, 60, 32], grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])
        self.predictor = KAN(layers_hidden=[32, 32, 1], grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])
        self._init_()

    def get_embedding(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(count))

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch=1,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr

class PINN_PDE_Discovery(nn.Module):
    """
    Fusion of the base PINN model and PDE_discovery.py:
    - Use sparse library regression (Phi * lambda_w) instead of dynamical_F neural network
    - Keep solution_u, DS weights, and all loss terms
    - Add interpretable PDE discovery capability
    """
    def __init__(self, args):
        super(PINN_PDE_Discovery, self).__init__()
        self.args = args
        if args.save_folder is not None and not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        log_dir = args.log_dir if args.save_folder is None else os.path.join(args.save_folder, args.log_dir)
        self.logger = get_logger(log_dir)
        self._save_args()
        requested_device = getattr(args, 'device', None)
        self.device = requested_device or ('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

        self.solution_u = Solution_u().to(self.device)
        
        # Regularization method: 'L1' (sparse L1 penalty), 'SINDy' (sequential hard thresholding / STLSQ), 'MLP' (neural network dynamics)
        self.reg_method = args.reg_method if hasattr(args, 'reg_method') else 'L1'
        self.sindy_threshold = args.sindy_threshold if hasattr(args, 'sindy_threshold') else 1e-2
        self.sindy_threshold_freq = args.sindy_threshold_freq if hasattr(args, 'sindy_threshold_freq') else 1
        self.sindy_iter = 0  # counter for thresholding frequency
        
        # Replace dynamical_F with sparse library regression
        self.PolyOrder = args.poly_order if hasattr(args, 'poly_order') else 2
        # Calculate total_terms based on input features
        # For battery: input is (x_features=16, t=1), u, u_x (16 dims), u_t (1 dim)
        # We build library from [u, u_x] which has 17 dims total
        num_state_vars = 17  # u (1) + u_x (16)
        # Polynomial terms: 1 + num_vars*P + higher order terms
        # For simplicity, estimate total_terms (will be set in build_library)
        self.total_terms = None  # Will be determined after first forward pass
        self.lambda_w = None  # Sparse coefficients, initialized after first batch
        self.lib_descr = []  # Library descriptions
        self.NonZeroMask_w = None  # Mask for sparsity
        
        self.optimizer1 = torch.optim.AdamW(self.solution_u.parameters(), lr=args.warmup_lr)
        # Optimizer for lambda_w will be added after initialization
        self.optimizer2 = None
        
        if self.reg_method == 'MLP':
            # MLP sparse regression: F = MLP(Phi), with the library Phi as input.
            # Library size is unknown until the first forward pass, so dynamical_F
            # is created lazily in init_dynamical_F() (same pattern as lambda_w).
            self.dynamical_F = None
            self.logger.info("reg_method=MLP: neural network regression on library Phi (lazy init, input dim = library size)")

        self.scheduler = LR_Scheduler(optimizer=self.optimizer1,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        self.best_model = None

        self.alpha = self.args.alpha
        self.beta = self.args.beta
        
        # L1 regularization for lambda_w (used in 'L1' and 'SINDy' modes; for SINDy it is an auxiliary penalty, set l1_coef=0 to disable)
        self.L1_coef = args.l1_coef if hasattr(args, 'l1_coef') else 1e-6

        # L0 regularization (continuous relaxation + optional hard thresholding)
        self.l0_coef = args.l0_coef if hasattr(args, 'l0_coef') else 1e-6
        self.l0_beta = args.l0_beta if hasattr(args, 'l0_beta') else 100.0
        self.l0_threshold = args.l0_threshold if hasattr(args, 'l0_threshold') else 0.0
        self.l0_threshold_freq = args.l0_threshold_freq if hasattr(args, 'l0_threshold_freq') else 1
        self.l0_iter = 0  # counter for L0 hard-thresholding frequency

        # Elastic Net regularization (L1 + L2)
        self.en_l1_coef = args.en_l1_coef if hasattr(args, 'en_l1_coef') else 1e-6
        self.en_l2_coef = args.en_l2_coef if hasattr(args, 'en_l2_coef') else 1e-6

        # Degradation-sensing weights use an exponential residual history.
        self.ds_weights1 = 0
        self.ds_weights2 = 0
        self.ds_eta = args.ds_eta if hasattr(args, 'ds_eta') else 0.99
        self.ds_decay = 1 - self.ds_eta
        self.init = 1

    def _save_args(self):
        if self.args.log_dir is not None:
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.critical(f"\t{k}:{v}")

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def build_library(self, u, u_x, x=None, t=None, data_description=['u'], PolyOrder=2):
        """
        Build candidate library from u, u_x, x, and t
        Similar to PDE_discovery.py but simplified for battery SOH case
        
        Args:
            u: (batch, 1) - SOH prediction
            u_x: (batch, 16) - gradients w.r.t. input features
            x: (batch, 16) - original input features (optional)
            t: (batch, 1) - time variable (optional)
            data_description: list of variable names
            PolyOrder: maximum polynomial order
        
        Returns:
            lib_terms: list of tensors, each is a candidate term
            lib_descr: list of strings describing each term
        """
        P = PolyOrder
        lib_poly = [torch.ones_like(u)]
        lib_poly_descr = ['1']
        
        # Polynomial terms of u
        for j in range(1, P + 1):
            lib_poly.append(u ** j)
            if j == 1:
                lib_poly_descr.append('u')
            else:
                lib_poly_descr.append(f"u**{j}")
        
        # Time terms (if provided)
        lib_t = []
        lib_t_descr = []
        if t is not None:
            for j in range(1, P + 1):
                lib_t.append(t ** j)
                if j == 1:
                    lib_t_descr.append('t')
                else:
                    lib_t_descr.append(f't**{j}')
        
        # Original input features (if provided)
        lib_x = []
        lib_x_descr = []
        if x is not None:
            for i in range(x.shape[1]):
                lib_x.append(x[:, i:i+1])
                lib_x_descr.append(f'x{i}')
        
        # Derivative terms (u_x has 16 components)
        lib_deri = []
        lib_deri_descr = []
        for i in range(u_x.shape[1]):
            lib_deri.append(u_x[:, i:i+1])
            lib_deri_descr.append(f'u_x{i}')
        
        # Multiplication of polynomials and derivatives
        # Skip i=0 (constant 1) to avoid redundancy with pure derivative terms
        lib_poly_deri = []
        lib_poly_deri_descr = []
        for i in range(1, len(lib_poly)):  # Start from 1 to skip "1 * derivative"
            for j in range(len(lib_deri)):
                lib_poly_deri.append(lib_poly[i] * lib_deri[j])
                lib_poly_deri_descr.append(lib_poly_descr[i] + lib_deri_descr[j])
        
        # Cross terms: u * t (if time is provided)
        lib_ut = []
        lib_ut_descr = []
        if t is not None:
            for i in range(1, len(lib_poly)):  # u, u^2, ...
                for j in range(len(lib_t)):  # t, t^2, ...
                    lib_ut.append(lib_poly[i] * lib_t[j])
                    lib_ut_descr.append(lib_poly_descr[i] + '*' + lib_t_descr[j])
        
        # Combine all terms
        all_terms = lib_poly + lib_t + lib_x + lib_deri + lib_poly_deri + lib_ut
        all_descr = lib_poly_descr + lib_t_descr + lib_x_descr + lib_deri_descr + lib_poly_deri_descr + lib_ut_descr
        
        return all_terms, all_descr

    def init_lambda_w(self, num_terms):
        """Initialize lambda_w after knowing the library size"""
        if self.lambda_w is None:
            self.total_terms = num_terms
            # Initialize with small random values
            self.lambda_w = nn.Parameter(
                torch.randn(num_terms, 1, device=self.device) * 0.01,
                requires_grad=True
            )
            self.NonZeroMask_w = torch.ones((num_terms, 1), device=self.device)
            
            # Create optimizer for lambda_w
            self.optimizer2 = torch.optim.AdamW(
                [self.lambda_w],
                lr=self.args.lr_F if hasattr(self.args, 'lr_F') else 1e-3
            )
            
            self.logger.info(f"Initialized lambda_w with {num_terms} library terms")

    def init_dynamical_F(self, num_terms):
        """Create dynamical_F lazily after knowing the library size (MLP mode).
        F = MLP(Phi), i.e. the MLP performs a (nonlinear) regression on the library Phi.
        """
        if self.dynamical_F is None:
            self.total_terms = num_terms
            self.dynamical_F = MLP(input_dim=num_terms, output_dim=1,
                                   layers_num=self.args.F_layers_num if hasattr(self.args, 'F_layers_num') else 3,
                                   hidden_dim=self.args.F_hidden_dim if hasattr(self.args, 'F_hidden_dim') else 60,
                                   droupout=0.2).to(self.device)
            self.optimizer2 = torch.optim.AdamW(
                self.dynamical_F.parameters(),
                lr=self.args.lr_F if hasattr(self.args, 'lr_F') else 1e-3)
            self.logger.info(f"reg_method=MLP: initialized dynamical_F with library input dim={num_terms}")

    def predict(self, xt):
        return self.solution_u(xt)

    def Test(self, testloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(testloader):
                x1 = x1.to(self.device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)

        return true_label, pred_label

    def Valid(self, validloader):
        self.eval()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for iter, (x1, _, y1, _) in enumerate(validloader):
                x1 = x1.to(self.device)
                u1 = self.predict(x1)
                true_label.append(y1)
                pred_label.append(u1.cpu().detach().numpy())
        pred_label = np.concatenate(pred_label, axis=0)
        true_label = np.concatenate(true_label, axis=0)
        mse = self.loss_func(torch.tensor(pred_label), torch.tensor(true_label))
        return mse.item()

    def forward(self, xt):
        """
        Forward pass with PDE discovery
        Replace dynamical_F with sparse library regression: F = Phi * lambda_w
        """
        xt.requires_grad = True
        x = xt[:, 0:-1]
        t = xt[:, -1:]

        u = self.solution_u(torch.cat((x, t), dim=1))

        u_t = grad(u.sum(), t,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]
        u_x = grad(u.sum(), x,
                   create_graph=True,
                   only_inputs=True,
                   allow_unused=True)[0]

        # Build library (needed by all modes: MLP uses Phi as input, L1/SINDy use Phi*lambda_w)
        lib_terms, self.lib_descr = self.build_library(u, u_x, x=x, t=t, PolyOrder=self.PolyOrder)
        Phi = torch.cat(lib_terms, dim=1)

        if self.reg_method == 'MLP':
            # MLP sparse regression: F = MLP(Phi), input is the library (num_terms dims)
            if self.dynamical_F is None:
                self.init_dynamical_F(len(lib_terms))
            F = self.dynamical_F(Phi)
        else:
            # Initialize lambda_w on first forward pass
            if self.lambda_w is None:
                self.init_lambda_w(len(lib_terms))
            
            # Compute F = Phi * (lambda_w * mask)
            F = torch.mm(Phi, self.lambda_w * self.NonZeroMask_w)

        # Cache Phi / u_t for the closed-form STLSQ step (reg_method == 'SINDy')
        if self.reg_method == 'SINDy':
            self.cache_Phi = Phi.detach()
            self.cache_u_t = u_t.detach()

        # PDE residual: f = u_t - F
        f = u_t - F
        
        return u, f

    def train_one_epoch(self, epoch, dataloader):
        self.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        loss4_meter = AverageMeter()  # L1 regularization loss
        
        for iter, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(self.device), x2.to(self.device), y1.to(self.device), y2.to(self.device)
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)

            # data loss
            loss1 = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)

            # PDE loss with degradation-sensing weights.
            ds_scale = 1 if self.init == 2 and iter == 0 else self.ds_eta
            r_norm1 = ds_scale * torch.abs(f1) / torch.max(torch.abs(f1))
            r_norm2 = ds_scale * torch.abs(f2) / torch.max(torch.abs(f2))
            
            if isinstance(self.ds_weights1, int):
                self.ds_weights1 = torch.zeros_like(r_norm1)
                self.ds_weights2 = torch.zeros_like(r_norm1)
                
            if r_norm1.size(0) < self.ds_weights1.size(0):
                padding = torch.zeros(self.ds_weights1.size(0) - r_norm1.size(0), 1, device=r_norm1.device)
                r_norm1 = torch.cat([r_norm1, padding], dim=0)
                r_norm2 = torch.cat([r_norm2, padding], dim=0)
            self.ds_weights1 = (self.ds_weights1 * self.ds_decay + r_norm1).detach()
            self.ds_weights2 = (self.ds_weights2 * self.ds_decay + r_norm2).detach()
            
            ds_weight1 = self.ds_weights1[:f1.size(0)]
            ds_weight2 = self.ds_weights2[:f2.size(0)]
            loss2 = 0.5 * torch.mean((ds_weight1 * f1) ** 2) + 0.5 * torch.mean((ds_weight2 * f2) ** 2)

            # physics loss  u2-u1<0, considering capacity regeneration
            loss3 = self.relu(torch.mul(u2 - u1, y1 - y2)).sum()
            
            # Regularization loss
            if self.reg_method == 'MLP':
                # MLP-based sparse regression: group-L1 penalty on the first layer of
                # dynamical_F, so that each candidate library term Phi_i is either
                # selected (non-zero input weights) or discarded (all weights -> 0).
                # W1 shape: (hidden_dim, num_terms); group norm over rows per input term.
                if self.dynamical_F is not None:
                    W1 = self.dynamical_F.net[0].weight  # first Linear layer
                    group_norm = torch.norm(W1, p=2, dim=0)  # per-library-term norm
                    loss4 = self.L1_coef * group_norm.sum()
                else:
                    loss4 = 0
            elif self.reg_method == 'L0':
                # L0 sparsity: differentiable relaxation of the l0 "norm"
                #   sum_i ( 2*sigmoid(beta*|lambda_i|) - 1 )  -> 0 as lambda->0, 1 as |lambda|->inf
                # Optionally combined with hard thresholding (see below) for exact zeros.
                if self.lambda_w is not None:
                    loss4 = self.l0_coef * torch.sum(
                        2 * torch.sigmoid(self.l0_beta * torch.abs(self.lambda_w)) - 1)
                else:
                    loss4 = 0
            elif self.reg_method == 'ElasticNet':
                # Elastic Net: L1 + L2 (Ridge) penalties
                if self.lambda_w is not None:
                    loss4 = (self.en_l1_coef * torch.norm(self.lambda_w, p=1)
                             + self.en_l2_coef * torch.norm(self.lambda_w, p=2) ** 2)
                else:
                    loss4 = 0
            else:  # 'L1' / 'SINDy'
                loss4 = self.L1_coef * torch.norm(self.lambda_w, p=1) if self.lambda_w is not None else 0

            # total loss
            loss = loss1 + self.alpha * loss2 + self.beta * loss3 + loss4

            self.optimizer1.zero_grad()
            if self.optimizer2 is not None:
                self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            if self.optimizer2 is not None:
                self.optimizer2.step()

            # SINDy: true STLSQ (sequential thresholded least squares).
            # Every sindy_threshold_freq iterations, refit lambda_w with a closed-form
            # least-squares solution on the *active* columns of Phi, then threshold.
            # This alternates gradient descent (via loss) with the analytic STLSQ step.
            if self.reg_method == 'SINDy' and self.lambda_w is not None:
                self.sindy_iter += 1
                if self.sindy_iter % self.sindy_threshold_freq == 0 and \
                        hasattr(self, 'cache_Phi') and self.cache_Phi is not None:
                    with torch.no_grad():
                        Phi_c = self.cache_Phi   # (batch, num_terms)
                        ut_c = self.cache_u_t    # (batch, 1)
                        # active set from current coefficients
                        active = (torch.abs(self.lambda_w) > self.sindy_threshold).squeeze(-1)
                        if active.sum() > 0:
                            Phi_a = Phi_c[:, active]          # (batch, n_active)
                            # closed-form least squares on active columns:
                            # coef_a = pinv(Phi_a) @ u_t
                            coef_a = torch.linalg.pinv(Phi_a) @ ut_c  # (n_active, 1)
                            new_lambda = torch.zeros_like(self.lambda_w)
                            new_lambda[active] = coef_a
                        else:
                            new_lambda = torch.zeros_like(self.lambda_w)
                        # threshold again
                        mask = (torch.abs(new_lambda) > self.sindy_threshold).float()
                        self.lambda_w.data = new_lambda * mask
                        self.NonZeroMask_w = mask

            # L0: hard thresholding (optional, when l0_threshold > 0).
            # Unlike SINDy (which refits with least squares), L0 only zeroes out
            # small coefficients via the mask, keeping the gradient path for the rest.
            if self.reg_method == 'L0' and self.lambda_w is not None and self.l0_threshold > 0:
                self.l0_iter += 1
                if self.l0_iter % self.l0_threshold_freq == 0:
                    with torch.no_grad():
                        mask = (torch.abs(self.lambda_w) > self.l0_threshold).float()
                        self.lambda_w.data = self.lambda_w.data * mask
                        self.NonZeroMask_w = mask

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            if isinstance(loss4, torch.Tensor):
                loss4_meter.update(loss4.item())

            if (iter + 1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.10f}, physics loss:{:.6f}, L1 loss:{:.6f}".format(
                    epoch, iter + 1, loss1, loss2, loss3, loss4 if isinstance(loss4, torch.Tensor) else 0))
                
        return loss1_meter.avg, loss2_meter.avg, loss3_meter.avg, loss4_meter.avg

    def get_sparse_coefficients(self):
        """Return (coef_vector, descr) for the current sparse regression.

        For 'L1'/'SINDy': lambda_w itself.
        For 'MLP': group-norm of the first-layer weights of dynamical_F, i.e.
        ||W1[:, i]||_2 per library term, which measures how strongly term i is used.
        Used by print_discovered_equation() and the term-selection-rate evaluation.
        """
        if self.reg_method == 'MLP':
            if self.dynamical_F is None or len(self.lib_descr) == 0:
                return None, None
            W1 = self.dynamical_F.net[0].weight.detach()  # (hidden_dim, num_terms)
            coef = torch.norm(W1, p=2, dim=0).cpu().numpy().flatten()  # per-term strength
            return coef, self.lib_descr
        if self.lambda_w is None or len(self.lib_descr) == 0:
            return None, None
        return self.lambda_w.detach().cpu().numpy().flatten(), self.lib_descr

    def print_discovered_equation(self):
        """Print the discovered PDE equation with feature importance analysis"""
        coef_np, descr = self.get_sparse_coefficients()
        if coef_np is None:
            self.logger.info("No equation discovered yet (coefficients not initialized)")
            return

        # Find non-zero terms
        disc_eq_temp = []
        for i in range(len(descr)):
            if abs(coef_np[i]) > 1e-6:  # Threshold for considering non-zero
                weight_value = coef_np[i]
                disc_eq_temp.append((abs(weight_value), f"{weight_value:.6f}*{descr[i]}"))
        
        # Sort by absolute value
        disc_eq_temp.sort(key=lambda x: x[0], reverse=True)
        
        # Take top 20 terms
        top_terms = [term for _, term in disc_eq_temp[:20]]
        
        if len(top_terms) > 0:
            disc_eq = ' + '.join(top_terms)
            self.logger.info(f"\nDiscovered PDE: du/dt = {disc_eq}\n")
            self.logger.info(f"Total non-zero terms: {len(disc_eq_temp)} out of {len(descr)}")
            
            # Feature importance analysis
            self.logger.info("\n" + "="*60)
            self.logger.info("FEATURE IMPORTANCE ANALYSIS")
            self.logger.info("="*60)
            
            # Extract feature importance for both x_i and u_xi terms
            x_importance = {}   # Original features
            ux_importance = {}  # Derivative features
            
            for i in range(len(descr)):
                term = descr[i]
                coef = abs(coef_np[i])
                
                import re
                # Match x_i terms (original features)
                if re.match(r'^x\d+$', term):  # Pure x terms like "x0", "x13"
                    match = re.search(r'x(\d+)', term)
                    if match:
                        feat_idx = int(match.group(1))
                        if feat_idx not in x_importance:
                            x_importance[feat_idx] = 0
                        x_importance[feat_idx] += coef
                
                # Match u_x_i terms (derivatives)
                if 'u_x' in term:
                    match = re.search(r'u_x(\d+)', term)
                    if match:
                        feat_idx = int(match.group(1))
                        if feat_idx not in ux_importance:
                            ux_importance[feat_idx] = 0
                        ux_importance[feat_idx] += coef
            
            # Display original feature importance
            if x_importance:
                sorted_x = sorted(x_importance.items(), key=lambda x: x[1], reverse=True)
                total_x = sum(x_importance.values())
                self.logger.info("\nOriginal Feature Importance (x_i directly in PDE):")
                for rank, (feat_idx, total_coef) in enumerate(sorted_x[:10], 1):
                    percentage = (total_coef / total_x) * 100 if total_x > 0 else 0
                    self.logger.info(f"  {rank}. x_{feat_idx}: |coef|={total_coef:.6f} ({percentage:.2f}%)")
            
            # Display derivative-based importance
            if ux_importance:
                sorted_ux = sorted(ux_importance.items(), key=lambda x: x[1], reverse=True)
                total_ux = sum(ux_importance.values())
                self.logger.info("\nDerivative-Based Importance (du/dx_i sensitivity):")
                for rank, (feat_idx, total_coef) in enumerate(sorted_ux[:10], 1):
                    percentage = (total_coef / total_ux) * 100 if total_ux > 0 else 0
                    self.logger.info(f"  {rank}. du/dx_{feat_idx}: |coef|={total_coef:.6f} ({percentage:.2f}%)")
            
            # Combined importance
            combined_importance = {}
            for feat_idx in set(list(x_importance.keys()) + list(ux_importance.keys())):
                combined_importance[feat_idx] = x_importance.get(feat_idx, 0) + ux_importance.get(feat_idx, 0)
            
            if combined_importance:
                sorted_combined = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
                total_combined = sum(combined_importance.values())
                self.logger.info("\nCombined Feature Importance (x_i + du/dx_i):")
                for rank, (feat_idx, total_coef) in enumerate(sorted_combined[:10], 1):
                    percentage = (total_coef / total_combined) * 100 if total_combined > 0 else 0
                    x_part = x_importance.get(feat_idx, 0)
                    ux_part = ux_importance.get(feat_idx, 0)
                    self.logger.info(f"  {rank}. x_{feat_idx}: {total_coef:.6f} ({percentage:.2f}%) [x:{x_part:.4f}, du/dx:{ux_part:.4f}]")
            
            self.logger.info("="*60 + "\n")
        else:
            self.logger.info("All coefficients are near zero")

    def Train(self, trainloader, testloader=None, validloader=None):
        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1, self.args.epochs + 1):
            early_stop += 1
            loss1, loss2, loss3, loss4 = self.train_one_epoch(e, trainloader)
            current_lr = self.scheduler.step()
            info = '[Train] epoch:{}, lr:{:.6f}, total loss:{:.6f}'.format(
                e, current_lr, loss1 + self.alpha * loss2 + self.beta * loss3 + loss4)
            info1 = "data loss:{:.6f}, PDE loss:{:.10f}, physics loss:{:.6f}, L1 loss:{:.6f}".format(
                loss1, loss2, loss3, loss4)
            info2 = "alpha:{:.6f}, beta:{:.6f}".format(self.alpha, self.beta)
            self.logger.info(info)
            self.logger.info(info1)
            self.logger.info(info2)
            
            # Print discovered equation every 10 epochs
            if e % 10 == 0:
                self.print_discovered_equation()
            
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                self.best_model = {
                    'solution_u': self.solution_u.state_dict(),
                    'lambda_w': self.lambda_w.detach().cpu() if self.lambda_w is not None else None,
                    'dynamical_F': self.dynamical_F.state_dict() if (self.reg_method == 'MLP' and self.dynamical_F is not None) else None,
                    'lib_descr': self.lib_descr,
                    'reg_method': self.reg_method
                }
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        
        # Print final discovered equation
        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL DISCOVERED EQUATION:")
        self.logger.info("="*80)
        self.print_discovered_equation()
        
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'model.pth'))
            # Save equation to text file
            coef_np, descr = self.get_sparse_coefficients()
            if coef_np is not None:
                with open(os.path.join(self.args.save_folder, 'discovered_equation.txt'), 'w') as f:
                    f.write(f"Discovered PDE Equation (reg_method={self.reg_method}):\n")
                    f.write("du/dt = \n")
                    for i in range(len(descr)):
                        if abs(coef_np[i]) > 1e-6:
                            f.write(f"  {coef_np[i]:.8f} * {descr[i]}\n")


if __name__ == "__main__":
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
        parser.add_argument('--batch', type=int, default=10, help='1,2,3')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')

        # Learning-rate scheduler settings.
        parser.add_argument('--epochs', type=int, default=100, help='epoch')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch')
        parser.add_argument('--warmup_lr', type=float, default=5e-4, help='warmup lr')
        parser.add_argument('--final_lr', type=float, default=1e-4, help='final lr')
        parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of lambda_w')
        parser.add_argument('--iter_per_epoch', type=int, default=1, help='iter per epoch')
        
        # PDE discovery related
        parser.add_argument('--poly_order', type=int, default=2, help='polynomial order for library')
        parser.add_argument('--reg_method', type=str, default='L1', choices=['L1', 'L0', 'ElasticNet', 'SINDy', 'MLP'],
                            help="regularization for dynamics: 'L1'=L1 sparse penalty, 'L0'=L0 sparsity (continuous relaxation + optional hard threshold), 'ElasticNet'=L1+L2 penalty, 'SINDy'=sequential hard thresholding (STLSQ), 'MLP'=neural network dynamical_F")
        parser.add_argument('--l1_coef', type=float, default=1e-6, help='L1 regularization coefficient for lambda_w (auxiliary penalty in SINDy mode)')
        parser.add_argument('--l0_coef', type=float, default=1e-6, help='L0 regularization coefficient (L0 mode)')
        parser.add_argument('--l0_beta', type=float, default=100.0, help='L0 relaxation steepness: 2*sigmoid(beta*|lambda|)-1')
        parser.add_argument('--l0_threshold', type=float, default=0.0, help='L0 hard-threshold: |lambda|<thr zeroed out every l0_threshold_freq steps (0=disabled)')
        parser.add_argument('--l0_threshold_freq', type=int, default=1, help='L0 hard-thresholding frequency (every N iterations)')
        parser.add_argument('--en_l1_coef', type=float, default=1e-6, help='ElasticNet L1 coefficient')
        parser.add_argument('--en_l2_coef', type=float, default=1e-6, help='ElasticNet L2 coefficient')
        parser.add_argument('--sindy_threshold', type=float, default=1e-2, help='SINDy threshold: coefficients below this are zeroed out')
        parser.add_argument('--sindy_threshold_freq', type=int, default=1, help='SINDy thresholding frequency (every N iterations)')
        parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of dynamical_F (MLP mode)')
        parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of dynamical_F (MLP mode)')

        parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
        parser.add_argument('--beta', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
        parser.add_argument('--ds_eta', type=float, default=0.99, help='degradation-sensing weight scale')
        parser.add_argument('--early_stop', type=int, default=20, help='early stop patience')

        parser.add_argument('--save_folder', type=str, default='./results/test_pde_discovery', help='save folder')
        parser.add_argument('--log_dir', type=str, default='log.txt', help='log dir')
        parser.add_argument('--gpu', type=str, default='1', help='gpu id')

        return parser.parse_args()


    args = get_args()
    pinn = PINN_PDE_Discovery(args)
    print("Solution_u network:")
    print(pinn.solution_u)
    count_parameters(pinn.solution_u)
    if pinn.reg_method == 'MLP':
        print("\nreg_method=MLP: neural network regression on library Phi (F = MLP(Phi))")
        print("dynamical_F is created lazily on the first forward pass (input dim = library size)")
    else:
        print("\nUsing sparse library regression instead of dynamical_F")
        print(f"Polynomial order: {pinn.PolyOrder}")
        print(f"L1 regularization: {pinn.L1_coef}")
        if pinn.reg_method == 'SINDy':
            print(f"SINDy threshold: {pinn.sindy_threshold}, frequency: {pinn.sindy_threshold_freq}")
