from dreal import *
import torch
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import time



exp_num = 81

folder_path = '{}/results/exp_{:02d}_keep_eg1'.format(str(Path(__file__).parent.parent), exp_num)


lyapunov_weights_file = 'trained_lyapunov_nn_iter_200.net'
controller_weights_file = 'trained_controller_nn_iter_200.net'
drift_vec_weights_file = 'trained_drift_vec_nn_iter_200.net'
control_vec_weights_file = 'trained_control_vec_nn_iter_200.net'

lyapunov_weights = torch.load(os.path.join(folder_path, lyapunov_weights_file))
controller_weights = torch.load(os.path.join(folder_path, controller_weights_file))
drift_vec_weights = torch.load(os.path.join(folder_path, drift_vec_weights_file))
control_vec_weights = torch.load(os.path.join(folder_path, control_vec_weights_file))

c_max = 0.11 # 0.0918

x1 = Variable("x1")
x2 = Variable("x2")
vars_ = np.array([x1, x2])
print(vars_)


# Lyapunov
print('####################################')
print('Weights in lyapunov nn:', lyapunov_weights.keys())
input_dim = 2

eps_quad = 1e-6
eps = 1e-6
matrix = lyapunov_weights['matrix'].numpy()
W_posdef_0 = lyapunov_weights['W_posdef_0'].numpy()
W_0 = lyapunov_weights['W_0'].numpy()
W_posdef_1 = lyapunov_weights['W_posdef_1'].numpy()


values_add = eps_quad*np.dot(vars_, vars_.T)

m = np.zeros((input_dim, input_dim))
tril_indices = np.tril_indices(n=input_dim, m=input_dim, k=0)
m[tril_indices[0], tril_indices[1]] = matrix
tmp = np.sum(np.power(np.dot(vars_, m),2))

values_quad = tmp + values_add

# grad_quad = 2*np.dot(vars_, np.dot(m, m.T) + eps_quad*np.eye(len(vars_)))

k0 = np.dot(W_posdef_0.T, W_posdef_0) + eps* np.eye(W_posdef_0.shape[1])
k0 = np.concatenate([k0, W_0], axis = 0)
z0 = np.dot(vars_, k0.T)
a0 = []
for i in range(len(z0)):
  a0.append(tanh(z0[i]))

# g_prime_0 = []
# for i in range(len(z0)):
#   g_prime_0.append(1-tanh(z0[i])**2)
# g_prime_0 = np.array(g_prime_0)
# g_prime_0 = g_prime_0[:,np.newaxis]
# der = g_prime_0 * k0

k1 = np.dot(W_posdef_1.T, W_posdef_1) + eps* np.eye(W_posdef_1.shape[1])
z1 = np.dot(a0, k1.T)
a1 = []
for i in range(len(z1)):
  a1.append(tanh(z1[i]))

# g_prime_1 = []
# for i in range(len(z1)):
#   g_prime_1.append(1-tanh(z1[i])**2)
# g_prime_1 = np.array(g_prime_1)
# g_prime_1 = g_prime_1[:,np.newaxis]
# der = np.dot(g_prime_1 * k1, der)

values_eth = np.sum(np.power(a1, 2))
# grad_eth = 2*np.dot(a1, der)

V = values_quad + values_eth
# grad_V = grad_quad + grad_eth

# Controller
print('####################################')
print('Weights in controller nn:', controller_weights.keys())

mul_param = controller_weights['mul_param'].numpy()
low_thresh_param = controller_weights['low_thresh_param'].numpy()
high_thresh_param = controller_weights['high_thresh_param'].numpy()
raw_low_slope_param = controller_weights['raw_low_slope_param'].numpy()
raw_high_slope_param = controller_weights['raw_high_slope_param'].numpy()
low_slope_param = mul_param * raw_low_slope_param
high_slope_param = mul_param * raw_high_slope_param
matrix = controller_weights['matrix'].numpy()

layer_0_weight = controller_weights['layer_0.weight'].numpy()
layer_0_bias = controller_weights['layer_0.bias'].numpy()
layer_1_weight = controller_weights['layer_1.weight'].numpy()
layer_1_bias = controller_weights['layer_1.bias'].numpy()
layer_2_weight = controller_weights['layer_2.weight'].numpy()
layer_2_bias = controller_weights['layer_2.bias'].numpy()

linear_part = np.dot(vars_, matrix.T)

out0 = np.dot(vars_, layer_0_weight.T) + layer_0_bias
a0 = []
for i in range(len(out0)):
  a0.append(tanh(out0[i]))

out1 = np.dot(a0, layer_1_weight.T) + layer_1_bias
a1 = []
for i in range(len(out1)):
  a1.append(tanh(out1[i]))

out2 = np.dot(a1, layer_2_weight.T) + layer_2_bias

out = linear_part.item() + out2.item()

# control = if_then_else(out>high_thresh_param, high_thresh_param + (out - high_thresh_param)*high_slope_param, 0) +\
#           if_then_else(out<low_thresh_param, low_thresh_param + (out - low_thresh_param)*low_slope_param, 0) +\
#           if_then_else(And(out>= low_thresh_param, out<= high_thresh_param), out, 0)

control = Min(Max(out, Expression(0) + low_thresh_param), Expression(0) + high_thresh_param) +\
            Max(out - high_thresh_param, Expression(0))*high_slope_param +\
            Min(out - low_thresh_param, Expression(0))*low_slope_param 

# Drift error
print('####################################')
print('Weights in drift nn:', drift_vec_weights.keys())
layer_0_weight = drift_vec_weights['layer_0.weight'].numpy()
layer_0_bias = drift_vec_weights['layer_0.bias'].numpy()
layer_1_weight = drift_vec_weights['layer_1.weight'].numpy()
layer_1_bias = drift_vec_weights['layer_1.bias'].numpy()
layer_2_weight = drift_vec_weights['layer_2.weight'].numpy()
layer_2_bias = drift_vec_weights['layer_2.bias'].numpy()

out0 = np.dot(vars_, layer_0_weight.T) + layer_0_bias
a0 = []
for i in range(len(out0)):
  a0.append(tanh(out0[i]))

out1 = np.dot(a0, layer_1_weight.T) + layer_1_bias
a1 = []
for i in range(len(out1)):
  a1.append(tanh(out1[i]))

out2 = np.dot(a1, layer_2_weight.T) + layer_2_bias

drift_error = out2.item()


# Control vector error
print('####################################')
print('Weights in control vec nn:',control_vec_weights)
W = control_vec_weights['W'].numpy()

control_vec_error = Expression(0) + W

# System dynamics
g = 9.81
m = 0.8
L = 0.4
b = 0.0
inertia = m * L**2
state_norm = (np.pi, np.pi)
action_norm = (10,)
normalization = [state_norm, action_norm]
normalization = [np.array(norm, dtype = np.float64) for norm in normalization]
inv_norm = [norm**-1 for norm in normalization]
Tx_inv, Tu_inv = map(np.diag, inv_norm)
Tx, Tu = map(np.diag, normalization)

state_denorm = np.dot(vars_, Tx)
control_denorm = np.dot(control, Tu)

angle, angular_velocity = state_denorm
x_ddot = (g / L * sin(angle) + control_denorm / inertia).item()
state_derivative = [angular_velocity, x_ddot]
state_derivative_norm = np.dot(state_derivative, Tx_inv)

nominal_closed_loop_dynamics = [state_derivative_norm[0],\
                                state_derivative_norm[1] + drift_error + control_vec_error*control]

# dot vnn
V_dot = V.Differentiate(x1) * nominal_closed_loop_dynamics[0] + V.Differentiate(x2) * nominal_closed_loop_dynamics[1]
# V_dot2 = grad_V[0] * nominal_closed_loop_dynamics[0] + grad_V[1] * nominal_closed_loop_dynamics[1]

print('####################################')
print('Plug in values to evaluate...')
env = {x1:0.1/np.sqrt(2), x2:0.1/np.sqrt(2)}
# env = {x1:-1, x2:-1}
# env = {x1:1, x2:1}
# env = {x1:0, x2:0}
t1 = time.time()
print(V.Evaluate(env))
t2 = time.time()
print('Time used:', t2-t1)

t1 = time.time()
print(control.Evaluate(env))
t2 = time.time()
print('Time used:', t2-t1)

t1 = time.time()
print(drift_error.Evaluate(env))
t2 = time.time()
print('Time used:', t2-t1)

t1 = time.time()
print(control_vec_error.Evaluate(env))
t2 = time.time()
print('Time used:', t2-t1)

t1 = time.time()
print(V_dot.Evaluate(env))
t2 = time.time()
print('Time used:', t2-t1)


print('####################################')
config = Config()
# config.use_polytope = True
config.precision = 1e-3
print('Precision:', config.precision)

print('c_max:', c_max)
cutoff_radius = 0.3/np.pi
print('cutoff_radius:', cutoff_radius)
ub = 1
bound = logical_and(x1*x1 + x2*x2> cutoff_radius**2, abs(x1) < ub, abs(x2) < ub, V <c_max)
condtion = logical_not(logical_imply(bound, V_dot + 0.1*(x1*x1 + x2*x2) <0))

start_time = time.time()
print('Started.')
result = CheckSatisfiability(condtion, config)
stop_time = time.time()
print(result)
print('Time used:', stop_time - start_time)
