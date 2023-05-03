import numpy as np
import sys
import math
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import copy
from time import perf_counter

import torch
import torch.optim as optim

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import mars
from mars import config
from mars.visualization import plot_roa_2D, plot_levelsets, plot_nested_roas
from mars.utils import load_controller_nn, print_no_newline, compute_nrows_ncolumns, str2bool
# from mars.roa_tools import initialize_roa, initialize_lyapunov_nn, initialize_lyapunov_quadratic, pretrain_lyapunov_nn, sample_around_roa, train_lyapunov_nn
from mars.roa_tools import *
from mars.dynamics_tools import *
from mars.utils import get_batch_grad, save_lyapunov_nn, load_lyapunov_nn, make_dataset_from_trajectories, save_dynamics_nn, load_dynamics_nn, save_controller_nn
from mars.controller_tools import *
from mars.parser_tools import getArgs

from examples.systems_config import all_systems 
from examples.example_utils import build_system, VanDerPol, InvertedPendulum, LyapunovNetwork, compute_roa_ct, balanced_class_weights, generate_trajectories, save_dict, load_dict


try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
import pickle
import os
import warnings
import random
import argparse
warnings.filterwarnings("ignore")

input_args_str = "\
--system pendulum1\
--dt 0.01\
--drift_vector_nn_sizes [16,16,1]\
--drift_vector_nn_activations ['tanh','tanh','identity']\
--control_vector_nn_sizes [1]\
--control_vector_nn_activations ['identity']\
--dynamics_batchsize 32\
--dynamics_loss_coeff 1\
--dynamics_pre_lr 0.1\
--dynamics_pre_iters 200\
--dynamics_train_lr 0.1\
--dynamics_train_iters 200\
--grid_resolution 100\
--repetition_use denoise\
--lyapunov_decrease_threshold 0.0\
--roa_gridsize 100\
--roa_pre_lr 1e-3\
--roa_pre_iters 10000\
--roa_pre_batchsize 32\
--roa_inner_iters 200\
--roa_outer_iters 100\
--roa_train_lr 1e-4\
--roa_lr_scheduler_step 20\
--roa_nn_structure quadratic\
--roa_nn_sizes [64,64,3]\
--roa_nn_activations ['tanh','tanh','identity']\
--roa_batchsize 32\
--roa_adaptive_level_multiplier True\
--roa_level_multiplier 3\
--roa_decrease_loss_coeff 1000.0\
--roa_decrease_alpha 0.1\
--roa_lipschitz_loss_coeff 0.01\
--roa_size_beta 0.0\
--roa_size_loss_coeff 0.00\
--controller_inner_iters 100\
--controller_outer_iters 2\
--controller_level_multiplier 2\
--controller_traj_length 10\
--controller_train_lr 1e-5\
--controller_batchsize 16\
--controller_train_slope True\
--verbose True\
--image_save_format pdf\
--exp_num 00\
--use_cuda False"

input_args_temp = input_args_str.split("--")
input_args = []
for ind, twins in enumerate(input_args_temp[1:]):
    a, b = twins.split(" ")
    a = "--{}".format(a)
    input_args.append(a)
    input_args.append(b)
args = getArgs(input_args)

device = config.device
print('Pytorch using device:', device)
exp_num = args.exp_num
results_dir = '{}/results/exp_{:02d}'.format(str(Path(__file__).parent.parent), exp_num)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

## Save hyper-parameters
# with open(os.path.join(results_dir, "00hyper_parameters.txt"), "w") as text_file:
#     input_args_temp = input_args_str.split("--")
#     for i in range(1, len(input_args_temp)):
#         text_file.write("--")
#         text_file.write(str(input_args_temp[i]))
#         text_file.write("\\")
#         text_file.write("\n")

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
resolution = args.grid_resolution # Number of states divisions each dimension

# Initialize system class and its linearization
system = build_system(system_properties, dt)
nominal_system = build_system(nominal_system_properties, dt)
# A, B = nominal_system.linearize_ct()
A, B = system.linearize_ct()
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
cutoff_radius    = 0.05
initial_safe_set = np.linalg.norm(grid.all_points, ord=2, axis=1) <= cutoff_radius

# LQR policy and its true ROA
Q = np.identity(state_dim).astype(config.np_dtype)  # state cost matrix
R = np.identity(action_dim).astype(config.np_dtype)  # action cost matrix
K_lqr, P_lqr = mars.utils.lqr(A, B, Q, R)
print("LQR matrix:", K_lqr)
# K = 0.4*K_lqr
K = np.array([[3, 1]], dtype=config.np_dtype)
print("K: ", K)

policy = mars.TrainableLinearController(-K, name='policy').to(device)
# bound = 0.2
# policy = mars.TrainableLinearControllerLooseThresh(-K, name='policy', \
#     args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
#         'high_slope':0.0, 'train_slope':args.controller_train_slope})

# Close loop dynamics
closed_loop_dynamics = lambda states: dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))
partial_closed_loop_dynamics = lambda states: nominal_dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))

horizon = 4000 # smaller tol requires longer horizon to give an accurate estimate of ROA
dt = 0.01
tol = 0.01 # how much close to origin must be x(T) to be considered as stable trajectory
end_states = np.array([[0,0],[0.01, 0],[-0.01, 0],[0.1, 0],[-0.1, 0], [1, 0], [-1, 0]], dtype= config.np_dtype)
trajectories = np.empty((end_states.shape[0], end_states.shape[1], horizon))
trajectories[:, :, 0] = end_states
with torch.no_grad():
    for t in range(1, horizon):
        trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1]).cpu().numpy()*dt + trajectories[:, :, t - 1]

    for i in range(end_states.shape[0]):
        trajectories[i, :, :] = np.matmul(Tx, trajectories[i, :, :])

def sinc(x):
    return np.sin(x)/x

# print('Tx:', Tx)
# print('Tu:', Tu)

for i in range(end_states.shape[0]):
    plt.plot(trajectories[i, 0, :], label = 'theta_'+ str(i))
    plt.plot(trajectories[i, 1, :], label = 'omega_'+ str(i))

plt.legend()
plt.savefig(os.path.join(results_dir, '00roa_test.{}'.format(args.image_save_format)), dpi=config.dpi)
m = system_properties.m
g = system_properties.g
L = system_properties.L
print('kp:', Tu.item()/(m*g*L)*np.matmul(K, np.linalg.inv(Tx))[0,0])
print('Theta 1:',  trajectories[1, 0, :][-1])
print('Theta 1 sinc:',  sinc(trajectories[1, 0, :][-1]))
print('Theta 2:',  trajectories[2, 0, :][-1])
print('Theta 2 sinc:',  sinc(trajectories[2, 0, :][-1]))
print('Theta 3:',  trajectories[3, 0, :][-1])
print('Theta 3 sinc:',  sinc(trajectories[1, 0, :][-1]))
print('Theta 4:',  trajectories[4, 0, :][-1])
print('Theta 4 sinc:',  sinc(trajectories[2, 0, :][-1]))
print('Theta 5:',  trajectories[5, 0, :][-1])
print('Theta 5 sinc:',  sinc(trajectories[1, 0, :][-1]))
print('Theta 6:',  trajectories[6, 0, :][-1])
print('Theta 6 sinc:',  sinc(trajectories[2, 0, :][-1]))