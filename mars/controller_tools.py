import numpy as np
import sys
sys.path.append('../mars/')
import mars
from mars.configuration import Configuration
config = Configuration()
del Configuration
from examples.example_utils import LyapunovNetwork
from mars.utils import get_batch_grad
from examples.example_utils import balanced_class_weights
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def initialize_controller():
    pass

def pretrain_controller_nn(target_set, policy_nn, target_policy, batchsize, n_iters, learning_rate, verbose=False, full_path=None):

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(policy_nn.parameters(), lr=learning_rate)
    loss_monitor = []
    for i in tqdm(range(n_iters)):
        optimizer.zero_grad()
        batch_inds = np.random.choice(target_set.shape[0], batchsize, replace=True)
        states = target_set[batch_inds]
        output_nn = policy_nn(states)
        output_pre = target_policy(states)
        loss = criterion(output_pre, output_nn)
        if verbose:
            print("loss: {}".format(loss))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_monitor.append(loss.detach().cpu().numpy())
    if full_path != None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        ax.plot(np.arange(0, n_iters).reshape(n_iters, 1), loss_monitor, linewidth=1)
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        # plt.ylim([0,0.02])
        plt.xticks(np.linspace(0, n_iters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(full_path, dpi=config.dpi)
        plt.close(fig)
    return policy_nn

def train_controller_SGD(target_set, lyapunov_nn, policy, closed_loop_dynamics, batchsize,
                      niters, policy_learning_rate,
                      alpha, beta, decrease_loss_coeff, Lipschitz_loss_coeff, size_loss_coeff,
                      fullpath_to_save_objectives=None, verbose=False):
   
    optimizer = optim.SGD(policy.parameters(), lr=policy_learning_rate)
    dot_vnn = lambda x: torch.sum(torch.mul(lyapunov_nn.grad_lyapunov_function(x), closed_loop_dynamics(x)),1)
    all_objectives_record = {"decrease": np.zeros(niters)}
    for ind_in in tqdm(range(niters)):
        optimizer.zero_grad()
        batch_inds = np.random.choice(target_set.shape[0], batchsize, replace=True)
        target_states_batch = target_set[batch_inds]
        decrease_loss = torch.max(dot_vnn(target_states_batch)\
            + alpha*torch.pow(torch.norm(torch.tensor(target_states_batch, dtype=config.ptdtype,\
             device=config.device), p=2, dim = 1),2) + 0.01, torch.tensor(0, dtype=config.ptdtype,\
                 device=config.device)).reshape(-1, 1)
        objective_decrease_condition = torch.mean(decrease_loss_coeff* decrease_loss)
        all_objectives_record["decrease"][ind_in] = objective_decrease_condition
        objective = objective_decrease_condition
        if verbose:
            print("Decrease_loss: {:8f}".format(objective_decrease_condition.detach().numpy()))   
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
    if fullpath_to_save_objectives is not None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        y_axis_values = all_objectives_record["decrease"].reshape(niters, 1)
        ax.plot(np.arange(0, niters).reshape(niters, 1), y_axis_values, linewidth=1)
        ax.legend(["Decrease"])
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        plt.xticks(np.linspace(0, niters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(fullpath_to_save_objectives, dpi=config.dpi)
        plt.close(fig)
    return policy
