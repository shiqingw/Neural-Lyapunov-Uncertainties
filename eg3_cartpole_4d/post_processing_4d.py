import os
import platform
if platform.system() == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    print("KMP_DUPLICATE_LIB_OK set to true in os.environ to avoid warning on Mac")
import numpy as np
import sys
import math
import matplotlib
import collections
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import copy
from time import perf_counter

import torch
import torch.optim as optim

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import mars
from mars import config
from mars.visualization import plot_nested_roas_3D, plot_nested_roas_3D_diagnostic
from mars.roa_tools import *
from mars.dynamics_tools import *
from mars.utils import get_batch_grad, save_lyapunov_nn, load_lyapunov_nn,\
     save_dynamics_nn, load_dynamics_nn, save_controller_nn, load_controller_nn, count_parameters
from mars.controller_tools import pretrain_controller_nn, train_controller_SGD
from mars.parser_tools import getArgs

from examples.systems_config import all_systems 
from examples.example_utils import build_system, VanDerPol, InvertedPendulum, LyapunovNetwork, compute_roa_ct, balanced_class_weights, generate_trajectories, save_dict, load_dict


try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
import pickle
import warnings
import random
import argparse
warnings.filterwarnings("ignore")

exp_num = 300

results_dir = '{}/results/exp_{:03d}_keep_eg3'.format(str(Path(__file__).parent.parent), exp_num)
# results_dir = '{}/results/exp_{:02d}_keep_eg3'.format(str(Path(__file__).parent.parent), exp_num)
# results_dir = '{}/results/exp_{:02d}'.format(str(Path(__file__).parent.parent), exp_num)

with open(os.path.join(results_dir, "00hyper_parameters.txt"), "r") as f:
    lines = f.readlines()

input_args = []
for line in lines:
    a, b = line.split(" ")
    b = b[0:-2]
    input_args.append(a)
    input_args.append(b)
args = getArgs(input_args)

device = config.device
print('Pytorch using device:', device)

# Set random seed
torch.manual_seed(0)
np.random.seed(0)
# Choosing the system
dt = args.dt   # sampling time
# System properties
system_properties = all_systems[args.system]
nominal_system_properties = all_systems[args.system + '_nominal']
state_dim     = system_properties.state_dim
action_dim    = system_properties.action_dim
state_limits  = np.array([[-1., 1.]] * state_dim)
action_limits = np.array([[-1., 1.]] * action_dim)
resolution = args.grid_resolution
# resolution = 200 


# Initialize system class and its linearization
system = build_system(system_properties, dt)
nominal_system = build_system(nominal_system_properties, dt)
A, B = nominal_system.linearize_ct()
dynamics = lambda x, y: system.ode_normalized(x, y)
nominal_dynamics = lambda x, y: nominal_system.ode_normalized(x, y)

# State grid
grid_limits = np.array([[-1., 1.], ] * state_dim)
grid = mars.GridWorld(grid_limits, resolution)
tau = np.sum(grid.unit_maxes) / 2
u_max = system.normalization[1].item()
Tx, Tu = map(np.diag, system.normalization)
Tx_inv, Tu_inv = map(np.diag, system.inv_norm)

# Set initial safe set as a ball around the origin (in normalized coordinates)
cutoff_radius    = 0.4
initial_safe_set = np.linalg.norm(grid.all_points, ord=2, axis=1) <= cutoff_radius

# LQR policy and its true ROA
Q = np.identity(state_dim).astype(config.np_dtype)  # state cost matrix
R = np.identity(action_dim).astype(config.np_dtype)  # action cost matrix
K_lqr, P_lqr = mars.utils.lqr(A, B, Q, R)
print("LQR matrix:", K_lqr)
K = K_lqr
# K = np.zeros((system_properties.state_dim, system_properties.action_dim), dtype = config.np_dtype)

# policy = mars.TrainableLinearController(-K, name='policy').to(device)
# bound = 0.3
# policy = mars.TrainableLinearControllerLooseThresh(-K, name='policy', \
#     args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
#         'high_slope':0.0, 'train_slope':args.controller_train_slope})

bound = 0.5 
controller_layer_dims = eval(args.controller_nn_sizes)
controller_layer_activations = eval(args.controller_nn_activations)

# policy = mars.NonLinearControllerLooseThresh(state_dim, controller_layer_dims,\
#     controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
#     args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
#     'high_slope':0.0, 'train_slope':args.controller_train_slope})

# policy = mars.NonLinearControllerLooseThreshWithLinearPart(state_dim, controller_layer_dims,\
#     -K, controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
#     args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
    # 'high_slope':0.0, 'train_slope':args.controller_train_slope})

policy = mars.NonLinearControllerLooseThreshWithLinearPartMulSlope(state_dim, controller_layer_dims,\
    -K, controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
    args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
    'high_slope':0.0, 'train_slope':args.controller_train_slope, 'slope_multiplier':args.controller_slope_multiplier})

closed_loop_dynamics = lambda states: dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))

#  Quadratic Lyapunov function for the LQR controller and its induced ROA
nominal_closed_loop_dynamics = None
L_pol = lambda x: np.linalg.norm(-K, 1) # # Policy (linear)
L_dyn = lambda x: np.linalg.norm(A, 1) + np.linalg.norm(B, 1) * L_pol(x) # Dynamics (linear approximation)
lyapunov_function = mars.QuadraticFunction(P_lqr)
grad_lyapunov_function = mars.LinearSystem((2 * P_lqr,))
dot_v_lqr = lambda x: torch.sum(torch.mul(grad_lyapunov_function(x), closed_loop_dynamics(x)),1)
L_v = lambda x: torch.norm(grad_lyapunov_function(x), p=1, dim=1, keepdim=True) # Lipschitz constant of the Lyapunov function
L_dv = lambda x: torch.norm(torch.tensor(2 * P_lqr, dtype=config.ptdtype, device=device))
lyapunov_lqr = mars.Lyapunov_CT(grid, lyapunov_function, grad_lyapunov_function,\
     closed_loop_dynamics, nominal_closed_loop_dynamics, L_dyn, L_v, L_dv, tau, initial_safe_set, decrease_thresh=0)
lyapunov_lqr.update_values()

# ###### NN Lyapunov ######
layer_dims = eval(args.roa_nn_sizes)
layer_activations = eval(args.roa_nn_activations)
decrease_thresh = args.lyapunov_decrease_threshold
# Initialize nn Lyapunov
L_pol = lambda x: np.linalg.norm(-K, 1) 
L_dyn = lambda x: np.linalg.norm(A, 1) + np.linalg.norm(B, 1) * L_pol(x) 
lyapunov_nn, grad_lyapunov_nn, dv_nn, L_v, tau = initialize_lyapunov_nn(grid, closed_loop_dynamics, None, L_dyn, 
            initial_safe_set, decrease_thresh, args.roa_nn_structure, state_dim, layer_dims, 
            layer_activations)

training_info = load_dict(os.path.join(results_dir, "training_info.npy"))
nominal_c_max_exp_values =  training_info["roa_info_nn"]["nominal_c_max_exp_values"]
nominal_c_max_exp_unconstrained_values =  training_info["roa_info_nn"]["nominal_c_max_exp_unconstrained_values"]


post_proc_info = {"grid_size":[], "roa_size":[], "forward_invariant_size":[],\
     "nn_forward_invariant_size":[], "lqr_forward_invariant_size":[],\
     "nn_roa_size":[], "lqr_roa_size":[]}

post_proc_info["nn_forward_invariant_size"] = copy.deepcopy(training_info["roa_info_nn"]["nominal_largest_exp_stable_set_sizes"])
post_proc_info["lqr_forward_invariant_size"] = copy.deepcopy(training_info["roa_info_lqr"]["nominal_largest_exp_stable_set_sizes"])
post_proc_info["nn_roa_size"] = copy.deepcopy(training_info["roa_info_nn"]["nominal_exp_stable_set_sizes"])
post_proc_info["lqr_roa_size"] = copy.deepcopy(training_info["roa_info_lqr"]["nominal_exp_stable_set_sizes"])

# Pretrained results
# policy = load_controller_nn(policy, full_path=os.path.join(results_dir, 'pretrained_controller_nn.net'))
policy = load_controller_nn(policy, full_path=os.path.join(results_dir, 'init_controller_nn.net'))
lyapunov_nn = load_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, 'pretrained_lyapunov_nn.net'))
horizon = 4000 
tol = 0.01 
roa_true, trajs = compute_roa_ct(grid, closed_loop_dynamics, dt, horizon, tol, no_traj=False)
forward_invariant = np.zeros_like(roa_true, dtype=bool)
tmp = np.zeros([sum(roa_true),], dtype=bool)
trajs_abs = np.abs(trajs)[roa_true]
for i in range(trajs_abs.shape[0]):
    traj_abs = trajs_abs[i,:,:]
    if np.max(traj_abs) <= 1:
        tmp[i] = True
forward_invariant[roa_true] = tmp
print("ROA: ", sum(roa_true))
print("Forward invariance: ", sum(forward_invariant))


post_proc_info["grid_size"].append(grid.num_points)
post_proc_info["roa_size"].append(sum(roa_true))
post_proc_info["forward_invariant_size"].append(sum(forward_invariant))

# selected_pics = [100]
# selected_pics = [x - 1 for x in selected_pics]
# for k in selected_pics:
for k in range(args.roa_outer_iters):
    print('Iteration {} out of {}'.format(k+1, args.roa_outer_iters))
    policy = load_controller_nn(policy, full_path=os.path.join(results_dir, 'trained_controller_nn_iter_{}.net'.format(k+1)))
    # lyapunov_nn = load_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, 'trained_lyapunov_nn_iter_{}.net'.format(k+1)))
    
    # Close loop dynamics and true region of attraction
    closed_loop_dynamics = lambda states: dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))

    roa_true, trajs = compute_roa_ct(grid, closed_loop_dynamics, dt, horizon, tol, no_traj=False)
    forward_invariant = np.zeros_like(roa_true, dtype=bool)
    tmp = np.zeros([sum(roa_true),], dtype=bool)
    trajs_abs = np.abs(trajs)[roa_true]
    for i in range(trajs_abs.shape[0]):
        traj_abs = trajs_abs[i,:,:]
        if np.max(traj_abs) <= 1:
            tmp[i] = True
    forward_invariant[roa_true] = tmp
    print("ROA: ", sum(roa_true))
    print("Forward invariance: ", sum(forward_invariant))
    post_proc_info["grid_size"].append(grid.num_points)
    post_proc_info["roa_size"].append(sum(roa_true))
    post_proc_info["forward_invariant_size"].append(sum(forward_invariant))

save_dict(post_proc_info, os.path.join(results_dir, "00post_proc_info.npy"))