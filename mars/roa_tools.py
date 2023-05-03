import numpy as np
import sys

from mars.functions import DiffSumOfTwo_ETH
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

def initialize_roa(grid, method='ball', cutoff_radius=0.0):
    """Takes a grid and mark the states within the ball of the specified radius as safe """
    initial_safe_set = np.linalg.norm(grid.all_points, ord=2, axis=1) <= cutoff_radius
    return initial_safe_set

def initialize_lyapunov_nn(grid, closed_loop_dynamics, nominal_closed_loop_dynamics, L_dyn, initial_safe_set, decrease_thresh, nn_structure, state_dim, layer_dims, layer_activations):
    """ Takes configuration of the neural net that acts as the lyapunov function 
    and outputs the initialized network"""
    lyapunov_function = LyapunovNetwork(state_dim, nn_structure, layer_dims, layer_activations, initializer=torch.nn.init.xavier_uniform)
    if nn_structure == "sum_of_two_eth":
        grad_lyapunov_nn = DiffSumOfTwo_ETH(lyapunov_function.net)
    else:
        grad_lyapunov_nn = lambda x: get_batch_grad(lyapunov_function, x)
    L_v = lambda x: torch.norm(grad_lyapunov_nn(x), p=1, dim=1, keepdim=True)
    L_dv = lambda x: get_batch_jacobian_norm(lyapunov_function, x)
    tau = np.sum(grid.unit_maxes) / 2
    lyapunov_nn = mars.Lyapunov_CT(grid, lyapunov_function, grad_lyapunov_nn, closed_loop_dynamics, nominal_closed_loop_dynamics, L_dyn, L_v, L_dv, tau, initial_safe_set, decrease_thresh=0)
    return lyapunov_nn, grad_lyapunov_nn, L_v, L_dv, tau

def initialize_lyapunov_quadratic(grid, P, closed_loop_dynamics, nominal_closed_loop_dynamics, L_dyn, initial_safe_set, decrease_thresh):
    lyapunov_function = mars.QuadraticFunction(P)
    grad_lyapunov = mars.LinearSystem((2 * P,))
    L_v = lambda x: torch.norm(grad_lyapunov(x), p=1, dim=1, keepdim=True)
    tau = np.sum(grid.unit_maxes) / 2
    L_dv = lambda x: torch.norm(torch.tensor(2 * P, dtype=config.ptdtype, device=config.device))
    lyapunov_pre = mars.Lyapunov_CT(grid, lyapunov_function, grad_lyapunov,
     closed_loop_dynamics, nominal_closed_loop_dynamics, L_dyn, L_v, L_dv, tau, initial_safe_set, decrease_thresh=0)
    return lyapunov_pre, grad_lyapunov, L_v, L_dv, tau

def pretrain_lyapunov_nn(grid, lyapunov_nn, target_lyapunov, batchsize, n_iters, learning_rate, verbose=False, full_path=None):
    """
    Takes initialized lyapunov_nn and pretrain it to match target_lyapunov. 
    target_lyapunov is usually a simple quadratic function.
    """
    ind_range = len(grid.all_points)
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(lyapunov_nn.lyapunov_function.net.parameters(), lr=learning_rate)
    loss_monitor = []
    for i in tqdm(range(n_iters)):
        optimizer.zero_grad()
        batch_inds = np.random.choice(ind_range, batchsize, replace=True)
        states = grid.all_points[batch_inds, :]
        output_nn = lyapunov_nn.lyapunov_function(states)
        output_pre = target_lyapunov.lyapunov_function(states)
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
        plt.xticks(np.linspace(0, n_iters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(full_path, dpi=config.dpi)
        plt.close(fig)
    return lyapunov_nn

def pretrain_lyapunov_nn_Adam(grid, lyapunov_nn, target_lyapunov, batchsize, n_iters, learning_rate, verbose=False, full_path=None):
    """
    Takes initialized lyapunov_nn and pretrain it to match target_lyapunov. 
    target_lyapunov is usually a simple quadratic function.
    """
    ind_range = len(grid.all_points)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(lyapunov_nn.lyapunov_function.net.parameters(), lr=learning_rate)
    loss_monitor = []
    trainloader = torch.utils.data.DataLoader(grid.all_points, batch_size=batchsize, shuffle=True)
    n_minibatch = len(trainloader)
    for k in tqdm(range(n_iters)):
        epoch_loss = 0
        for i, states in enumerate(trainloader):
            optimizer.zero_grad()
            batch_inds = np.random.choice(ind_range, batchsize, replace=True)
            states = grid.all_points[batch_inds, :]
            output_nn = lyapunov_nn.lyapunov_function(states)
            output_pre = target_lyapunov.lyapunov_function(states)
            loss = criterion(output_pre, output_nn)
            if verbose:
                print("loss: {}".format(loss))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epoch_loss += loss.detach().cpu().numpy()
        loss_monitor.append(epoch_loss/n_minibatch)
    if full_path != None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        ax.plot(np.arange(0, n_iters).reshape(n_iters, 1), loss_monitor, linewidth=1)
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        plt.xticks(np.linspace(0, n_iters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(full_path, dpi=config.dpi)
        plt.close(fig)
    return lyapunov_nn

def train_largest_ROA_Adam(target_set, lyapunov_nn, policy, closed_loop_dynamics, batchsize,
                      niters, lyapunov_learning_rate, policy_learning_rate,
                      alpha, decrease_offset, beta, decrease_loss_coeff, Lipschitz_loss_coeff, size_loss_coeff,
                      fullpath_to_save_objectives=None, verbose=False, optimizer=None, lr_scheduler=None):
    if optimizer == None:
        optimizer = optim.Adam([
                {'params': lyapunov_nn.grad_lyapunov_function.parameters(), 'lr': lyapunov_learning_rate},
                {'params': policy.parameters(), 'lr': policy_learning_rate}
            ])
    trainloader = torch.utils.data.DataLoader(target_set, batch_size=batchsize, shuffle=True)
    n_minibatch = len(trainloader)
    dot_vnn = lambda x: torch.sum(torch.mul(lyapunov_nn.grad_lyapunov_function(x), closed_loop_dynamics(x)),1)
    all_objectives_record = {"decrease":np.zeros(niters), "Lipschitz":np.zeros(niters), "size":np.zeros(niters)}
    offset = decrease_offset
    for k in tqdm(range(niters)):
        decrease_epoch_loss = 0
        lipschitze_epoch_loss = 0
        for ind_in, target_states_batch in enumerate(trainloader):
            # Training step
            optimizer.zero_grad()
            
            decrease_loss = torch.max(dot_vnn(target_states_batch)\
                + alpha*torch.pow(torch.norm(torch.tensor(target_states_batch, dtype=config.ptdtype,\
                device=config.device), p=2, dim = 1),2) + offset, torch.tensor(0, dtype=config.ptdtype,\
                    device=config.device)).reshape(-1, 1)
            
            Lipschitz_loss = torch.norm(lyapunov_nn.grad_lyapunov_function(target_states_batch), p=2, dim=1)
            
            # size_loss = torch.add(-beta * torch.pow(torch.norm(torch.tensor(target_states_batch, dtype=config.ptdtype,\
            #      device=config.device), p=2, dim = 1),2), lyapunov_nn.lyapunov_function(target_states_batch))

            objective_decrease_condition = torch.mean(decrease_loss_coeff * decrease_loss)
            objective_Lipschitz = torch.mean(Lipschitz_loss_coeff * Lipschitz_loss)
            objective = objective_decrease_condition + objective_Lipschitz
            objective.backward()
            optimizer.step()
            decrease_epoch_loss = decrease_epoch_loss + objective_decrease_condition.item()
            lipschitze_epoch_loss = lipschitze_epoch_loss + objective_Lipschitz.item()

        all_objectives_record["decrease"][k], all_objectives_record["Lipschitz"][k], \
                = decrease_epoch_loss/n_minibatch, lipschitze_epoch_loss/n_minibatch

    if lr_scheduler != None:
        lr_scheduler.step()

    if fullpath_to_save_objectives is not None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        y_axis_values = np.concatenate([all_objectives_record["decrease"].reshape(niters, 1),\
             all_objectives_record["Lipschitz"].reshape(niters, 1)], 1)
        ax.plot(np.arange(0, niters).reshape(niters, 1), y_axis_values, linewidth=1)
        ax.legend(["Decrease", "Lipschitz"])
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        plt.xticks(np.linspace(0, niters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(fullpath_to_save_objectives, dpi=config.dpi)
        plt.close(fig)
    return lyapunov_nn


def train_largest_ROA_SGD(target_set, lyapunov_nn, policy, closed_loop_dynamics, batchsize,
                      niters, lyapunov_learning_rate, policy_learning_rate,
                      alpha, beta, decrease_loss_coeff, Lipschitz_loss_coeff, size_loss_coeff,
                      fullpath_to_save_objectives=None, verbose=False, optimizer=None, lr_scheduler=None):
    if optimizer == None:
        optimizer = optim.SGD([
                {'params': lyapunov_nn.lyapunov_function.net.parameters(), 'lr': lyapunov_learning_rate},
                {'params': policy.parameters(), 'lr': policy_learning_rate}
            ])

    Lipschitz_est_func = lambda x, e: (lyapunov_nn.lyapunov_function(x + e) - 
                                       lyapunov_nn.lyapunov_function(x)) / e
   
    dot_vnn = lambda x: torch.sum(torch.mul(lyapunov_nn.grad_lyapunov_function(x), closed_loop_dynamics(x)),1)
    all_objectives_record = {"decrease":np.zeros(niters), "Lipschitz":np.zeros(niters), "size":np.zeros(niters)}
    for ind_in in tqdm(range(niters)):
        # Training step
        optimizer.zero_grad()
        batch_inds = np.random.choice(target_set.shape[0], batchsize, replace=True)
        target_states_batch = target_set[batch_inds]
        
        decrease_loss = torch.max(dot_vnn(target_states_batch)\
            + alpha*torch.pow(torch.norm(torch.tensor(target_states_batch, dtype=config.ptdtype,\
             device=config.device), p=2, dim = 1),2) + 0.01, torch.tensor(0, dtype=config.ptdtype,\
                 device=config.device)).reshape(-1, 1)
        
        e = 0.1 # the direction to compute Lipschitz constant
        Lipschitz_loss = torch.norm(Lipschitz_est_func(target_states_batch, e), p=2, dim=1)
        
        size_loss = torch.add(-beta * torch.pow(torch.norm(torch.tensor(target_states_batch, dtype=config.ptdtype,\
             device=config.device), p=2, dim = 1),2), lyapunov_nn.lyapunov_function(target_states_batch))

        objective_decrease_condition = torch.mean(decrease_loss_coeff * decrease_loss)
        objective_Lipschitz = torch.mean(Lipschitz_loss_coeff * Lipschitz_loss)
        objective_size = torch.mean(size_loss_coeff * size_loss)
        objective = objective_decrease_condition + objective_Lipschitz + objective_size
        
        all_objectives_record["decrease"][ind_in], all_objectives_record["Lipschitz"][ind_in], \
            all_objectives_record["size"][ind_in]  = objective_decrease_condition, \
                objective_Lipschitz, objective_size

        if verbose:
            print("Decrease_loss:{:8f} \n Lipschitz loss:{:8f} \n Size loss:{:8f}"\
                .format(objective_decrease_condition.detach().numpy(), \
                    objective_Lipschitz.detach().numpy(), objective_size.detach().numpy()))   
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        if lr_scheduler != None:
            lr_scheduler.step()
    if fullpath_to_save_objectives is not None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        y_axis_values = np.concatenate([all_objectives_record["size"].reshape(niters, 1), all_objectives_record["decrease"].reshape(niters, 1), all_objectives_record["Lipschitz"].reshape(niters, 1)], 1)
        ax.plot(np.arange(0, niters).reshape(niters, 1), y_axis_values, linewidth=1)
        ax.legend(["Size", "Decrease", "Lipschitz"])
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        plt.xticks(np.linspace(0, niters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(fullpath_to_save_objectives, dpi=config.dpi)
        plt.close(fig)
    return lyapunov_nn

def train_largest_ROA_under_c_SGD(target_set, target_below_c, lyapunov_nn, policy, closed_loop_dynamics, batchsize,
                      niters, lyapunov_learning_rate, policy_learning_rate,
                      alpha, c, decrease_loss_coeff, Lipschitz_loss_coeff, classification_loss_coeff,
                      fullpath_to_save_objectives=None, verbose=False, optimizer=None, lr_scheduler=None):
    if optimizer == None:
        optimizer = optim.SGD([
                {'params': lyapunov_nn.lyapunov_function.net.parameters(), 'lr': lyapunov_learning_rate},
                {'params': policy.parameters(), 'lr': policy_learning_rate}
            ])
    
    # optimizer = optim.SGD(lyapunov_nn.lyapunov_function.net.parameters(),  lyapunov_learning_rate)

    Lipschitz_est_func = lambda x, e: (lyapunov_nn.lyapunov_function(x + e) - 
                                       lyapunov_nn.lyapunov_function(x)) / e
   
    dot_vnn = lambda x: torch.sum(torch.mul(lyapunov_nn.grad_lyapunov_function(x), closed_loop_dynamics(x)),1)
    all_objectives_record = {"decrease":np.zeros(niters), "Lipschitz":np.zeros(niters), "classification":np.zeros(niters)}
    for ind_in in tqdm(range(niters)):
        # Training step
        optimizer.zero_grad()
        batch_inds = np.random.choice(target_set.shape[0], batchsize, replace=True)
        target_states_batch = target_set[batch_inds]
        batch_inds = np.random.choice(target_below_c.shape[0], batchsize, replace=True)
        target_below_c_batch = target_below_c[batch_inds]
        
        decrease_loss = torch.max(dot_vnn(target_states_batch)\
            + alpha*torch.pow(torch.norm(torch.tensor(target_states_batch, dtype=config.ptdtype,\
             device=config.device), p=2, dim = 1),2) + 0.01, torch.tensor(0, dtype=config.ptdtype,\
                 device=config.device)).reshape(-1, 1)
        
        e = 0.1 # the direction to compute Lipschitz constant
        Lipschitz_loss = torch.norm(Lipschitz_est_func(target_states_batch, e), p=2, dim=1)
        
        classification_loss = torch.max(lyapunov_nn.lyapunov_function(target_below_c_batch) - c, \
            torch.tensor(0, dtype=config.ptdtype, device=config.device)).reshape(-1,1)

        objective_decrease_condition = torch.mean(decrease_loss_coeff * decrease_loss)
        objective_Lipschitz = torch.mean(Lipschitz_loss_coeff * Lipschitz_loss)
        objective_classification = torch.mean(classification_loss_coeff * classification_loss)
        objective = objective_decrease_condition + objective_Lipschitz + objective_classification
        
        all_objectives_record["decrease"][ind_in], all_objectives_record["Lipschitz"][ind_in], \
            all_objectives_record["classification"][ind_in]  = objective_decrease_condition, \
                objective_Lipschitz, objective_classification

        if verbose:
            print("Decrease_loss:{:8f} \n Lipschitz loss:{:8f} \n Size loss:{:8f}"\
                .format(objective_decrease_condition.detach().numpy(), \
                    objective_Lipschitz.detach().numpy(), objective_classification.detach().numpy()))   
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        if lr_scheduler != None:
            lr_scheduler.step()
    if fullpath_to_save_objectives is not None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        y_axis_values = np.concatenate([all_objectives_record["classification"].reshape(niters, 1), all_objectives_record["decrease"].reshape(niters, 1), all_objectives_record["Lipschitz"].reshape(niters, 1)], 1)
        ax.plot(np.arange(0, niters).reshape(niters, 1), y_axis_values, linewidth=1)
        ax.legend(["Classification", "Decrease", "Lipschitz"])
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        plt.xticks(np.linspace(0, niters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(fullpath_to_save_objectives, dpi=config.dpi)
        plt.close(fig)
    return lyapunov_nn

def train_lyapunov_SGD(target_set, lyapunov_nn, closed_loop_dynamics, batchsize,
                      niters, lyapunov_learning_rate,
                      alpha, beta, decrease_loss_coeff, Lipschitz_loss_coeff, size_loss_coeff,
                      fullpath_to_save_objectives=None, verbose=False):

    
    optimizer = optim.SGD([
               {'params': lyapunov_nn.lyapunov_function.net.parameters(), 'lr': lyapunov_learning_rate}
           ])
    # optimizer = optim.SGD(lyapunov_nn.lyapunov_function.net.parameters(),  lyapunov_learning_rate)

    Lipschitz_est_func = lambda x, e: (lyapunov_nn.lyapunov_function(x + e) - 
                                       lyapunov_nn.lyapunov_function(x)) / e
   
    dot_vnn = lambda x: torch.sum(torch.mul(lyapunov_nn.grad_lyapunov_function(x), closed_loop_dynamics(x)),1)
    all_objectives_record = {"decrease":np.zeros(niters), "Lipschitz":np.zeros(niters), "size":np.zeros(niters)}
    for ind_in in tqdm(range(niters)):
        # Training step
        optimizer.zero_grad()
        batch_inds = np.random.choice(target_set.shape[0], batchsize, replace=True)
        target_states_batch = target_set[batch_inds]
        
        decrease_loss = torch.max(dot_vnn(target_states_batch)\
            + alpha*torch.pow(torch.norm(torch.tensor(target_states_batch, dtype=config.ptdtype,\
             device=config.device), p=2, dim = 1),2) + 0.01, torch.tensor(0, dtype=config.ptdtype,\
                 device=config.device)).reshape(-1, 1)
        
        e = 0.1 # the direction to compute Lipschitz constant
        Lipschitz_loss = torch.norm(Lipschitz_est_func(target_states_batch, e), p=2, dim=1)
        
        size_loss = torch.add(-beta * torch.pow(torch.norm(torch.tensor(target_states_batch, dtype=config.ptdtype,\
             device=config.device), p=2, dim = 1),2), lyapunov_nn.lyapunov_function(target_states_batch))

        objective_decrease_condition = torch.mean(decrease_loss_coeff * decrease_loss)
        objective_Lipschitz = torch.mean(Lipschitz_loss_coeff * Lipschitz_loss)
        objective_size = torch.mean(size_loss_coeff * size_loss)
        objective = objective_decrease_condition + objective_Lipschitz + objective_size
        
        all_objectives_record["decrease"][ind_in], all_objectives_record["Lipschitz"][ind_in], \
            all_objectives_record["size"][ind_in]  = objective_decrease_condition, \
                objective_Lipschitz, objective_size

        if verbose:
            print("Decrease_loss:{:8f} \n Lipschitz loss:{:8f} \n Size loss:{:8f}"\
                .format(objective_decrease_condition.detach().numpy(), \
                    objective_Lipschitz.detach().numpy(), objective_size.detach().numpy()))   
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
    if fullpath_to_save_objectives is not None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        y_axis_values = np.concatenate([all_objectives_record["size"].reshape(niters, 1), all_objectives_record["decrease"].reshape(niters, 1), all_objectives_record["Lipschitz"].reshape(niters, 1)], 1)
        ax.plot(np.arange(0, niters).reshape(niters, 1), y_axis_values, linewidth=1)
        ax.legend(["Size", "Decrease", "Lipschitz"])
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        plt.xticks(np.linspace(0, niters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(fullpath_to_save_objectives, dpi=config.dpi)
        plt.close(fig)
    return lyapunov_nn

def sample_around_roa(lyapunov_nn, expansion_factor, n_samples, method="gap"):
    """
    Takes the lyapunov_nn and expand its safe levelset such by the expansion_factor and choose
    samples from this expansion gap.
    
    
    Parameters
    ----------
    lyapunov_nn : lyapunov_nn class
    expansion_factor: a real number larger than 1.0
    n_samples: number of intended initial states to be chosen
    method: {"gap", etc}, a method to choose the samples. 
            "gap: chooses initial states from the gap"
    
    Returns
    -------
    A binary mask of the chosen indices. The size is the same as
    len(lyapunov_nn.discretization.all_points) and the where it is all False
    and only True for the chosen states.
    
    """

    grid = lyapunov_nn.discretization
    c = lyapunov_nn.c_max.detach().numpy()
    idx_small = lyapunov_nn.values.detach().numpy().ravel() <= c
    idx_big = lyapunov_nn.values.detach().numpy().ravel() <= c * expansion_factor
    idx_gap = np.logical_and(idx_big, ~idx_small)
    chosen_numerical_indx = np.random.choice(np.where(idx_gap == True)[0], n_samples, replace=False)
    idx_chosen = np.zeros_like(idx_gap) == 1
    idx_chosen[chosen_numerical_indx] = True
    return idx_chosen


def sample_blindly(grid, n_samples, method="uniform", rad=None):
    """
    Takes the grid and choose the initial states from that grid based on the provided  method.
    Note that the Lyapunov function or ROA is not used here for sampling.
    
    
    Parameters
    ----------
    grid : state discretization
    n_samples: number of intended initial states to be chosen
    method: {"uniform"}, determines the method of sampling from the grid.
    
    Returns
    -------
    A binary mask of the chosen indices. The size is the same as
    len(grid.all_points) where it is all False
    and only True for the chosen states.
    
    """
    if method == "uniform":
        chosen_numerical_indx = np.random.choice(grid.nindex, n_samples, replace=False)
        idx_chosen = np.zeros(grid.nindex) == 1
        idx_chosen[chosen_numerical_indx] = True
    elif method == "ball":
        idx_feasible = np.linalg.norm(grid.all_points, 2, axis=1) <= rad
        if sum(idx_feasible) < n_samples:
            raise ValueError("The number of chosen samples is larger than the size of the feasible set")
        else:
            feasible_numerical_idx = np.where(idx_feasible)[0]
            chosen_numerical_indx = np.random.choice(feasible_numerical_idx, n_samples, replace=False)
            idx_chosen = np.zeros(grid.nindex) == 1
            idx_chosen[chosen_numerical_indx] = True
    return idx_chosen

def find_exp_stable_region(grid, lyapunov_nn, closed_loop_dynamics, alpha, full_path=None):
    all_points = grid.all_points
    dot_vnn = lambda x: torch.sum(torch.mul(lyapunov_nn.grad_lyapunov_function(x), closed_loop_dynamics(x)), dim=1)
    decrease = torch.add(dot_vnn(all_points), \
            alpha*torch.pow(torch.norm(torch.tensor(all_points, dtype=config.ptdtype,\
            device=config.device), p=2, dim = 1),2)).reshape(-1, 1)
    if full_path != None:
        with open(full_path, 'w') as f:
            ll = decrease.detach().cpu().numpy()
            for i in range(len(ll)):
                f.write("{} \n".format(ll[i]))
    exp_stable_region = decrease <= 0
    exp_stable_region = exp_stable_region.detach().cpu().numpy().ravel()
    return exp_stable_region
    
