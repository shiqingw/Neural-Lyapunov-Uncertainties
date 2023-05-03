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

exp_num = 43

results_dir = '{}/results/exp_{:02d}_keep_eg2'.format(str(Path(__file__).parent.parent), exp_num)
# results_dir = '{}/results/exp_{:02d}'.format(str(Path(__file__).parent.parent), exp_num)

#################### Draw roas ####################
print("#################### Constrained RoA ####################")
training_info = load_dict(os.path.join(results_dir, "training_info.npy"))
grid_size = training_info["grid_size"]
roa_info = training_info["roa_info_nn"]
true_roa_sizes = np.array(roa_info["true_roa_sizes"])
true_largest_exp_stable_set_sizes = np.array(roa_info["true_largest_exp_stable_set_sizes"])
nominal_largest_exp_stable_set_sizes = np.array(roa_info["nominal_largest_exp_stable_set_sizes"])

true_roa_ratio_nn = true_roa_sizes/grid_size
true_largest_exp_stable_ratio_nn = true_largest_exp_stable_set_sizes/grid_size
nominal_largest_exp_stable_ratio_nn = nominal_largest_exp_stable_set_sizes/grid_size

post_proc_info = load_dict(os.path.join(results_dir, "00post_proc_info.npy"))
forward_invariant_size = np.array(post_proc_info["forward_invariant_size"])

forward_invariant_ratio = forward_invariant_size/grid_size
print("forward_invariant_ratio", forward_invariant_ratio[0], forward_invariant_ratio[-1])
print("true_roa_ratio_nn: ", true_roa_ratio_nn[0], true_roa_ratio_nn[-1])
print("true_largest_exp_stable_ratio_nn: ", true_largest_exp_stable_ratio_nn[-1])
print("nominal_largest_exp_stable_ratio_nn: ", nominal_largest_exp_stable_ratio_nn[-1])

roa_info = training_info["roa_info_lqr"]
true_largest_exp_stable_set_sizes = np.array(roa_info["true_largest_exp_stable_set_sizes"])
nominal_largest_exp_stable_set_sizes = np.array(roa_info["nominal_largest_exp_stable_set_sizes"])

true_largest_exp_stable_ratio_lqr = true_largest_exp_stable_set_sizes/grid_size
nominal_largest_exp_stable_ratio_lqr = nominal_largest_exp_stable_set_sizes/grid_size

print("true_largest_exp_stable_ratio_lqr: ", true_largest_exp_stable_ratio_lqr[-1])
print("nominal_largest_exp_stable_ratio_lqr: ", nominal_largest_exp_stable_ratio_lqr[-1])

fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
# fig = plt.figure(figsize=(50, 10), dpi=config.dpi, frameon=False)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
labelsize =50
ticksize = 40
legendsize = 30
lw = 6

plt.plot(true_roa_ratio_nn, linewidth = lw, linestyle = 'dashed', label = "RoA")
plt.plot(forward_invariant_ratio, linewidth = lw, linestyle = 'dotted', label = "Forward Invariant RoA")
plt.plot(nominal_largest_exp_stable_ratio_nn, linewidth = lw, linestyle = 'solid', label = "Estimated RoA (Ours)")
plt.plot(true_largest_exp_stable_ratio_lqr, linewidth = lw, linestyle = 'dashdot', label = "Estimated RoA (LQR)")


plt.legend(loc="best",fontsize=legendsize)
# plt.xlabel(r"$\rm{Iteration}$", fontsize=fontsize)
# plt.ylabel(r"$\rm{Ratio}$", fontsize=fontsize)
plt.xlabel("Iteration", fontsize=labelsize)
plt.ylabel("Ratios", fontsize=labelsize)
# plt.ylim([0,1.1])
# plt.xticks(np.arange(0,100,2))
plt.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=20)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '00roa_ratio.eps'), dpi=config.dpi)
plt.clf()

# assert(False)

#################### Draw trajs ####################

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
cutoff_radius    = 0.05
initial_safe_set = np.linalg.norm(grid.all_points, ord=2, axis=1) <= cutoff_radius

# LQR policy and its true ROA
K = np.zeros((system_properties.action_dim, system_properties.state_dim), dtype = config.np_dtype)

bound = 0.1 
controller_layer_dims = eval(args.controller_nn_sizes)
controller_layer_activations = eval(args.controller_nn_activations)
# policy = mars.NonLinearControllerLooseThresh(state_dim, controller_layer_dims,\
#     controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
#     args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
#     'high_slope':0.0, 'train_slope':args.controller_train_slope})

policy = mars.NonLinearControllerLooseThreshWithLinearPart(state_dim, controller_layer_dims,\
    -K, controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
    args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
    'high_slope':0.0, 'train_slope':args.controller_train_slope})

policy = load_controller_nn(policy, full_path=os.path.join(results_dir, "trained_controller_nn_iter_{}.net".format(args.roa_outer_iters)))


# Close loop dynamics and true region of attraction
closed_loop_dynamics = lambda states: dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))

horizon = 4000 
tol = 0.01 
roa_true = compute_roa_ct(grid, closed_loop_dynamics, dt, horizon, tol, no_traj=True) # True ROA with LQR policy
grid_size = grid.num_points

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
lyapunov_nn = load_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, "trained_lyapunov_nn_iter_{}.net".format(args.roa_outer_iters)))
lyapunov_nn.update_values()
lyapunov_nn.update_safe_set('true', roa_true)
lyapunov_nn.update_exp_stable_set(args.roa_decrease_alpha, 'true', roa_true)

training_info = load_dict(os.path.join(results_dir, "training_info.npy"))
c = training_info["roa_info_nn"]["nominal_c_max_values"][-1]
print(c)
c1 = lyapunov_nn.c_max_exp_true
print(c1)

print("Determining the limit points")
ind_higher = lyapunov_nn.values.detach().cpu().numpy().ravel() <= c
ind_lower = lyapunov_nn.values.detach().cpu().numpy().ravel() <= c - 0.001
ind = np.logical_and(ind_higher, ~ind_lower)
print(np.sum(ind))


################## Plot specific trajs ##################
## Close loop dynamics
plot_limits = np.dot(Tx, grid_limits)
closed_loop_dynamics = lambda states: dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))

horizon = 800 
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
plt.ylabel(r"$x_1$", fontsize=labelsize)
plt.ylim(plot_limits[0])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '00traj_test_x1.eps'), dpi=config.dpi)
plt.clf()

for i in range(end_states.shape[0]):
    plt.plot(time, trajectories_denormalized[i, 1, :], linewidth = lw, label = "Trajectory " + str(i+1))
# plt.legend(fontsize = legend_fontsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.xlabel(r"time (s)", fontsize=labelsize)
plt.ylabel(r"$x_2$", fontsize=labelsize)
plt.ylim(plot_limits[1])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '00traj_test_x2.eps'), dpi=config.dpi)
plt.clf()

for i in range(end_states.shape[0]):
    plt.plot(time, trajectories_denormalized[i, 2, :], linewidth = lw, label = "Trajectory " + str(i+1))
# plt.legend(fontsize = legend_fontsize)
plt.xticks(fontsize = ticksize)
plt.yticks(fontsize = ticksize)
plt.xlabel(r"time (s)", fontsize=labelsize)
plt.ylabel(r"$x_3$", fontsize=labelsize)
plt.ylim(plot_limits[2])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '00traj_test_x3.eps'), dpi=config.dpi)
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
plt.savefig(os.path.join(results_dir, '00traj_test_norm.eps'), dpi=config.dpi)
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
plt.savefig(os.path.join(results_dir, '00traj_test_u.eps'), dpi=config.dpi)
