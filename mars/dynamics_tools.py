import numpy as np
import sys
sys.path.append('../mars/')
import mars
from mars.configuration import Configuration
config = Configuration()
del Configuration
from examples.example_utils import LyapunovNetwork
from mars.utils import get_batch_grad, get_batch_jacobian_norm
from examples.example_utils import balanced_class_weights
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def train_dynamics_sample_in_batch_SGD(target_set, dot_v, target_dot_v, drift_vec_nn, 
    control_vec_nn, coefficient, batchsize, n_iters, learning_rate, full_path=None, print_grad=False):
    params = list(drift_vec_nn.parameters()) + list(control_vec_nn.parameters())
    optimizer = optim.SGD(params, lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    loss = []
    for i in tqdm(range(n_iters)):
        optimizer.zero_grad()
        batch_inds = np.random.choice(target_set.shape[0], batchsize, replace=True)
        target_states_batch =target_set[batch_inds]
        target = target_dot_v(target_states_batch)
        # print(torch.max(torch.absolute(target)))
        output = dot_v(target_states_batch)
        # print(torch.max(torch.absolute(output)))
        objective = coefficient * loss_function(target, output)
        # print(objective)
        loss.append(objective.item())
        optimizer.zero_grad()
        objective.backward()
        if print_grad:
            for param in params:
                print(param.grad.norm())
        optimizer.step()
    if full_path is not None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        y_axis_values = np.array(loss).reshape(n_iters, 1)
        ax.plot(np.arange(0, n_iters).reshape(n_iters, 1), y_axis_values, linewidth=1)
        ax.legend(["MSE loss"])
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        plt.xticks(np.linspace(0, n_iters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(full_path, dpi=config.dpi)
        plt.close(fig)
    return 

def train_dynamics_sample_in_batch_Adam(target_set, dot_v, target_dot_v, drift_vec_nn, 
    control_vec_nn, coefficient, batchsize, n_iters, learning_rate, full_path=None, print_grad=False):
    params = list(drift_vec_nn.parameters()) + list(control_vec_nn.parameters())
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=0.01)
    trainloader = torch.utils.data.DataLoader(target_set, batch_size=batchsize, shuffle=True)
    n_minibatch = len(trainloader)
    loss_function = torch.nn.MSELoss()
    loss = []
    for k in tqdm(range(n_iters)):
        epoch_loss = 0
        for i, target_states_batch in enumerate(trainloader):
            optimizer.zero_grad()
            target = target_dot_v(target_states_batch)
            output = dot_v(target_states_batch)
            objective = coefficient * loss_function(target, output)
            epoch_loss = epoch_loss + objective.item()
            optimizer.zero_grad()
            objective.backward()
            if print_grad:
                for param in params:
                    print(param.grad.norm())
            optimizer.step()
        loss.append(epoch_loss/n_minibatch)
    if full_path is not None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        y_axis_values = np.array(loss).reshape(n_iters, 1)
        ax.plot(np.arange(0, n_iters).reshape(n_iters, 1), y_axis_values, linewidth=1)
        ax.legend(["MSE loss"])
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        plt.xticks(np.linspace(0, n_iters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(full_path, dpi=config.dpi)
        plt.close(fig)
    return 

def train_dynamics_SGD(grid, dot_v, target_dot_v, drift_vec_nn, control_vec_nn, batchsize, n_iters, learning_rate, full_path=None):
    params = list(drift_vec_nn.parameters()) + list(control_vec_nn.parameters())
    optimizer = optim.SGD(params, lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    loss = []
    for i in tqdm(range(n_iters)):
        optimizer.zero_grad()
        batch_inds = np.random.choice(grid.all_points.shape[0], batchsize, replace=True)
        target_states_batch =grid.all_points[batch_inds]
        target = target_dot_v(target_states_batch)
        output = dot_v(target_states_batch)
        objective = loss_function(target, output)
        loss.append(objective.item())
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
    if full_path is not None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        y_axis_values = np.array(loss).reshape(n_iters, 1)
        ax.plot(np.arange(0, n_iters).reshape(n_iters, 1), y_axis_values, linewidth=1)
        ax.legend(["MSE loss"])
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        plt.xticks(np.linspace(0, n_iters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(full_path, dpi=config.dpi)
        plt.close(fig)
    return 

def train_dynamics(grid, dot_v, target_dot_v, drift_vec_nn, control_vec_nn, batchsize, n_iters, learning_rate, full_path=None):
    params = list(drift_vec_nn.parameters()) + list(control_vec_nn.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    loss = []
    for i in tqdm(range(n_iters)):
        optimizer.zero_grad()
        points = grid.all_points
        target = target_dot_v(points)
        output = dot_v(points)
        objective = loss_function(target, output)
        loss.append(objective.item())
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
    if full_path is not None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        y_axis_values = np.array(loss).reshape(n_iters, 1)
        ax.plot(np.arange(0, n_iters).reshape(n_iters, 1), y_axis_values, linewidth=1)
        ax.legend(["MSE loss"])
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        plt.xticks(np.linspace(0, n_iters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(full_path, dpi=config.dpi)
        plt.close(fig)
    return 