import numpy as np
from matplotlib import pyplot as plt
import os
import mars
from mars import config

exp_num = 00
results_dir = './results/exp_{:02d}'.format(exp_num)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


def dynamics(x1, x2, x3, K_lqr):
    dx1 = x2
    dx2 = x3
    dx3 = (x1*x1) + np.matmul(-K_lqr, np.array([[x1],[x2],[x3]], dtype=config.np_dtype)).item()
    return dx1, dx2, dx3

x1 = [1]
x2 = [1]
x3 = [1]
horizon = 4000
dt = 0.01

# LQR policy and its true ROA
Q = np.identity(3).astype(config.np_dtype)  # state cost matrix
R = np.identity(1).astype(config.np_dtype)  # action cost matrix
A = np.array([[0, 1, 0],[0, 0, 1],[2, 0, 0]], dtype = config.np_dtype)
B = np.array([[0],[0],[1]], dtype = config.np_dtype)

K_lqr, P_lqr = mars.utils.lqr(A, B, Q, R)
print("LQR matrix:", K_lqr)


for i in range(horizon):
    dx1, dx2, dx3 = dynamics(x1[-1], x2[-1], x3[-1], K_lqr)
    x1.append(x1[-1]+dx1*dt)
    x2.append(x2[-1]+dx2*dt)
    x3.append(x3[-1]+dx3*dt)

plt.plot(x1, label = "x1")
plt.plot(x2, label = "x2")
plt.plot(x3, label = "x3")
plt.legend()
plt.savefig(os.path.join(results_dir, '00traj_test.png'), dpi=config.dpi)

