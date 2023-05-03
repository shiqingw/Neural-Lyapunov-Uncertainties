import os
import platform
if platform.system() == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    print("KMP_DUPLICATE_LIB_OK set to true in os.environ to avoid warning on Mac")
import numpy as np
from matplotlib import pyplot as plt
import pickle 
import torch
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import mars
from mars import config
from mars.visualization import plot_traj_on_levelset, plot_phase_portrait
from mars.utils import load_controller_nn, print_no_newline, compute_nrows_ncolumns, str2bool
from mars.dynamics_tools import *
from mars.utils import load_lyapunov_nn, load_dynamics_nn
from mars.roa_tools import initialize_lyapunov_nn
from mars.parser_tools import getArgs

from examples.systems_config import all_systems 
from examples.example_utils import build_system, VanDerPol, InvertedPendulum, LyapunovNetwork, compute_roa_ct, balanced_class_weights, generate_trajectories, save_dict, load_dict


import warnings
warnings.filterwarnings("ignore")

exp_num = 300

results_dir = '{}/results/exp_{:03d}_keep_eg3'.format(str(Path(__file__).parent.parent), exp_num)
# results_dir = '{}/results/exp_{:02d}_keep_eg3'.format(str(Path(__file__).parent.parent), exp_num)
# results_dir = '{}/results/exp_{:02d}_eg3'.format(str(Path(__file__).parent.parent), exp_num)
# results_dir = '{}/results/exp_{:02d}'.format(str(Path(__file__).parent.parent), exp_num)



# #################### Draw trajs ####################

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
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
# Choosing the system
dt = args.dt   # sampling time
# System properties
system_properties = all_systems[args.system]
nominal_system_properties = all_systems[args.system + '_nominal']
perturbed_system_properties = all_systems[args.system + '_perturbed']
state_dim     = system_properties.state_dim
action_dim    = system_properties.action_dim
state_limits  = np.array([[-1., 1.]] * state_dim)
action_limits = np.array([[-1., 1.]] * action_dim)
resolution = args.grid_resolution

# Initialize system class and its linearization
system = build_system(system_properties, dt)
nominal_system = build_system(nominal_system_properties, dt)
perturbed_system = build_system(perturbed_system_properties, dt)
A, B = nominal_system.linearize_ct()
dynamics = lambda x, y: system.ode_normalized(x, y)
nominal_dynamics = lambda x, y: nominal_system.ode_normalized(x, y)
perturbed_dynamics = lambda x, y: perturbed_system.ode_normalized(x, y)

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
K = np.zeros((system_properties.action_dim, system_properties.state_dim), dtype = config.np_dtype)

controller_layer_dims = eval(args.controller_nn_sizes)
controller_layer_activations = eval(args.controller_nn_activations)
bound = 0.5 
# policy = mars.NonLinearControllerLooseThresh(state_dim, controller_layer_dims,\
#     controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
#     args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
#     'high_slope':0.0, 'train_slope':args.controller_train_slope})

# policy = mars.NonLinearControllerLooseThreshWithLinearPart(state_dim, controller_layer_dims,\
#     -K, controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
#     args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
#     'high_slope':0.0, 'train_slope':args.controller_train_slope})

policy = mars.NonLinearControllerLooseThreshWithLinearPartMulSlope(state_dim, controller_layer_dims,\
    -K, controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
    args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
    'high_slope':0.0, 'train_slope':args.controller_train_slope, 'slope_multiplier':args.controller_slope_multiplier})

policy = load_controller_nn(policy, full_path=os.path.join(results_dir, "trained_controller_nn_iter_{}.net".format(args.roa_outer_iters)))
# print(policy.low_slope_param, policy.high_slope_param)
print(policy.mul_low_slope_param, policy.mul_high_slope_param)
# assert(False)

# Close loop dynamics and true region of attraction
closed_loop_dynamics = lambda states: perturbed_dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))


# horizon = 4000 
# tol = 0.01 
# roa_true = compute_roa_ct(grid, closed_loop_dynamics, dt, horizon, tol, no_traj=True) # True ROA with LQR policy
grid_size = grid.num_points

# ###### NN Lyapunov ######
layer_dims = eval(args.roa_nn_sizes)
layer_activations = eval(args.roa_nn_activations)
decrease_thresh = args.lyapunov_decrease_threshold
# Initialize nn Lyapunov
L_pol = lambda x: np.linalg.norm(-K, 1) 
L_dyn = lambda x: np.linalg.norm(A, 1) + np.linalg.norm(B, 1) * L_pol(x) 
lyapunov_nn, grad_lyapunov_nn, dv_nn, L_v, tau = initialize_lyapunov_nn(grid, None, None, L_dyn, 
            initial_safe_set, decrease_thresh, args.roa_nn_structure, state_dim, layer_dims, 
            layer_activations)
lyapunov_nn = load_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, "trained_lyapunov_nn_iter_{}.net".format(args.roa_outer_iters)))
lyapunov_nn.update_values()
# lyapunov_nn.update_safe_set('true', roa_true)
# lyapunov_nn.update_exp_stable_set(args.roa_decrease_alpha, 'true', roa_true)
# assert(False)

training_info = load_dict(os.path.join(results_dir, "training_info.npy"))
c = training_info["roa_info_nn"]["nominal_c_max_values"][-1]
print(c)

print("Determining the limit points")
ind_higher = lyapunov_nn.values.detach().cpu().numpy().ravel() <= c
ind_lower = lyapunov_nn.values.detach().cpu().numpy().ravel() <= c - 0.002
ind = np.logical_and(ind_higher, ~ind_lower)
print(np.sum(ind))


# ################## Plot specific trajs ##################
# ## Close loop dynamics
plot_limits = np.dot(Tx, grid_limits)
# print(plot_limits)
# assert(False)

horizon = 3500 
dt = 0.01
time = [i*dt for i in range(horizon)]
target_set = grid.all_points[ind]
batch_inds = np.random.choice(target_set.shape[0], 10, replace=False)
end_states = target_set[batch_inds]

trajectories = np.empty((end_states.shape[0], end_states.shape[1], horizon))
trajectories[:, :, 0] = end_states
trajectories_denormalized = np.zeros_like(trajectories)
with torch.no_grad():
    for t in range(1, horizon):
        trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1]).cpu().numpy()*dt + trajectories[:, :, t - 1]

    for i in range(end_states.shape[0]):
        trajectories_denormalized[i, :, :] = np.matmul(Tx, trajectories[i, :, :])

print(trajectories.shape)
labelsize = 50
ticksize = 40
lw = 4
fig = plt.figure(figsize=(10, 7), dpi=config.dpi, frameon=False)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

for i in range(end_states.shape[0]):
    plt.plot(time, trajectories_denormalized[i, 0, :], linewidth = lw, label = "Trajectory " + str(i+1))
# plt.legend(fontsize = legend_fontsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.xlabel(r"time (s)", fontsize=labelsize)
plt.ylabel(r"$x$", fontsize=labelsize)
plt.ylim(plot_limits[0])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '00perturbed_traj_test_x.eps'), dpi=config.dpi)
plt.clf()

for i in range(end_states.shape[0]):
    plt.plot(time, trajectories_denormalized[i, 1, :], linewidth = lw, label = "Trajectory " + str(i+1))
# plt.legend(fontsize = legend_fontsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.xlabel(r"time (s)", fontsize=labelsize)
plt.ylabel(r"$\theta$", fontsize=labelsize)
plt.ylim(plot_limits[1])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '00perturbed_traj_test_theta.eps'), dpi=config.dpi)
plt.clf()

for i in range(end_states.shape[0]):
    plt.plot(time, trajectories_denormalized[i, 2, :], linewidth = lw, label = "Trajectory " + str(i+1))
# plt.legend(fontsize = legend_fontsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.xlabel(r"time (s)", fontsize=labelsize)
plt.ylabel(r"$v$", fontsize=labelsize)
plt.ylim(plot_limits[2])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '00perturbed_traj_test_v.eps'), dpi=config.dpi)
plt.clf()

for i in range(end_states.shape[0]):
    plt.plot(time, trajectories_denormalized[i, 3, :], linewidth = lw, label = "Trajectory " + str(i+1))
# plt.legend(fontsize = legend_fontsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.xlabel(r"time (s)", fontsize=labelsize)
plt.ylabel(r"$\omega$", fontsize=labelsize)
plt.ylim(plot_limits[3])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '00perturbed_traj_test_omega.eps'), dpi=config.dpi)
plt.clf()

for i in range(end_states.shape[0]):
    norm = np.linalg.norm(trajectories_denormalized[i, :, :], ord=2, axis=0)
    plt.plot(time, norm, linewidth = lw, label = "Trajectory " + str(i+1))
# plt.legend(fontsize = legend_fontsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.xlabel(r"time (s)", fontsize=labelsize)
plt.ylabel(r"norm of states", fontsize=labelsize)
# plt.ylim(plot_limits[3])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '00perturbed_traj_test_norm.eps'), dpi=config.dpi)
plt.clf()

for i in range(end_states.shape[0]):
    states = trajectories[i, :, :].T
    u = policy(states).detach().cpu().numpy()
    u = np.matmul(u, Tu).ravel()
    plt.plot(time, u, linewidth = lw, label = "Trajectory " + str(i+1))
# plt.legend(fontsize = legend_fontsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.xlabel(r"time (s)", fontsize=labelsize)
plt.ylabel(r"$u$", fontsize=labelsize)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '00perturbed_traj_test_u.eps'), dpi=config.dpi)
