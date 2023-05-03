import matplotlib
from matplotlib import scale
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import numpy as np
from .configuration import Configuration
config = Configuration()
del Configuration
from mars.utils import binary_cmap, get_number_of_rows_and_columns
import os
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn as nn
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def generate_trajectories(states_init, closed_loop_dynamics, dt, horizon):
    if isinstance(states_init, np.ndarray):
        states_init = torch.tensor(np.copy(states_init), dtype=config.ptdtype, device=config.device)
    nindex = states_init.shape[0]
    ndim = states_init.shape[1]
    
    trajectories = torch.zeros((nindex, ndim, horizon+1), dtype=config.ptdtype, device=config.device)
    trajectories[:, :, 0] = states_init
    
    with torch.no_grad():
        for t in range(1, horizon+1):
            trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1])*dt + trajectories[:, :, t - 1]
    return trajectories[:,:, 0:-1]

def add_arrow(line, position=None, direction='right', color=None, size = None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()
    if size is None:
        size = line.get_linewidth()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        start_ind = int(len(xdata)*0.001)
    else:
        # start_ind = np.argmin(np.absolute(xdata - position))
        start_ind = int(len(xdata)* position)
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(color= color),
        size=size*20
    )

def add_point(line, position=None, color=None, size = None):
    """
    add a point to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()
    if size is None:
        size = line.get_linewidth()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        start_ind = 0
    else:
        # start_ind = np.argmin(np.absolute(xdata - position))
        start_ind = int(len(xdata)* position)

    line.axes.annotate('', # this is the text
                (xdata[start_ind], ydata[start_ind]), # these are the coordinates to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,0), # distance from text to points (x,y)
                ha='center', # horizontal alignment can be left, right or center
                size = size*10)

def plot_roa_2D(roa, plot_limits, plot_labels, full_path):
    """ take roa data as a 2D matrix and save
    the plot in full_path

    Parameter
    -------------
    roa: N x N grid
    plot_limits: [(xmin, xmax), (ymin, ymax)]
    full_path: path to save the image file
    plot_labels: [label of x axis, label of y axis]
    """
    if roa.ndim != 2: 
        raise ValueError('Can only plot 2D RoA!')
    dir_path = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    if np.sum(roa) != roa.shape[0]*roa.shape[1]:
        ax = plt.subplot(111)
        color=[None]
        color[0] = (0, 158/255, 115/255)       # ROA - bluish-green
        # True ROA
        z = roa
        alpha = 1
        ax.set_xlabel(plot_labels[0], fontsize=50)
        ax.set_ylabel(plot_labels[1], fontsize=50)
        ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(color[0],), linewidths=1)
        ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[0]), alpha=alpha)
        ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
        ax.set_aspect(plot_limits[0][1]/plot_limits[1][1])
    else:
        ax = plt.subplot(111)
        color=[None]
        color[0] = (0, 158/255, 115/255)       # ROA - bluish-green
        # True ROA
        z = roa
        alpha = 1
        ax.set_xlabel(plot_labels[0], fontsize=50)
        ax.set_ylabel(plot_labels[1], fontsize=50)
        ax.contourf(z.T, origin='lower', extent=plot_limits.ravel(), colors=(color[0],), linewidths=1)
        ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[0]), alpha=alpha)
        ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
        ax.set_aspect(plot_limits[0][1]/plot_limits[1][1])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(full_path, dpi=config.dpi)
    # plt.close(fig)

def plot_nested_roas(grid_size, ind_small, ind_big, plot_limits, plot_labels, ind_gap_stable=None, ind_exp_stable=None, ind_true_roa=None, full_path=None):
    """ take a grid and the indices for small and big ROAs and plot them overlaid
    the plot in full_path

    Parameter
    -------------
    grid_size: size of the encompassing grid for both ROAs
    ind_small: binary ndarray vector, indices for the points of the inner ROA
    ind_big: binary ndarray vector, indices for points of the outer ROA
    ind_points, ind_gap_stable, ind_true_roa: optional, a binary vector with the same size of prod(grid_size).
            Each group of points are plotted with different colour to differentiate betweent the nature of points.
    plot_limits: [(xmin, xmax), (ymin, ymax)]
    full_path: path to save the image file
    """
    if full_path is not None: 
        dir_path = os.path.dirname(full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(111)
    color=[None, None, None, None, None]
    color[0] = (80/255, 100/255, 250/255)       # ROA - bluish-green
    color[1] = (158/255, 0, 115/255)
    color[2] = (158/255, 115/255, 0)
    color[3] = (0, 158/255, 115/255)
    color[4] = (255/255, 0, 0)

    # Draw from big areas to small areas to avoid obstructing 
    nindex = grid_size[0] * grid_size[1]
    
    if ind_true_roa is not None:
        roa_true = np.zeros(nindex, dtype=bool)
        roa_true[ind_true_roa] = True
        roa_true = roa_true.reshape(grid_size)
        roa_true[:,0] = False
        roa_true[:,-1] = False
        roa_true[0] = False
        roa_true[-1] = False
        ax.imshow(roa_true.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[4]), alpha=0.4)
        # ax.contour(roa_true.T, origin='lower', extent=plot_limits.ravel(), colors=(color[4],), linewidths=0.05, alpha=0.3)

    if ind_exp_stable is not None:
        exp_stable = np.zeros(nindex, dtype=bool)
        exp_stable[ind_exp_stable] = True
        exp_stable = exp_stable.reshape(grid_size)
        exp_stable[:,0] = False
        exp_stable[:,-1] = False
        exp_stable[0] = False
        exp_stable[-1] = False
        ax.imshow(exp_stable.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[3]), alpha=0.5)
        # ax.contour(exp_stable.T, origin='lower', extent=plot_limits.ravel(), colors=(color[3],), linewidths=0.05, alpha=0.5)


    if ind_gap_stable is not None:
        roa_gap_stable = np.zeros(nindex, dtype=bool)
        roa_gap_stable[ind_gap_stable] = True
        roa_gap_stable = roa_gap_stable.reshape(grid_size)
        roa_gap_stable[:,0] = False
        roa_gap_stable[:,-1] = False
        roa_gap_stable[0] = False
        roa_gap_stable[-1] = False
        ax.imshow(roa_gap_stable.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[2]), alpha=0.6)
        # ax.contour(roa_gap_stable.T, origin='lower', extent=plot_limits.ravel(), colors=(color[2],), linewidths=0.05, alpha=0.6)

    roa_big = np.zeros(nindex, dtype=bool)
    roa_big[ind_big] = True
    roa_big = roa_big.reshape(grid_size)
    roa_big[:,0] = False
    roa_big[:,-1] = False
    roa_big[0] = False
    roa_big[-1] = False
    ax.imshow(roa_big.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[1]), alpha=0.7)
    # ax.contour(roa_big.T, origin='lower', extent=plot_limits.ravel(), colors=(color[1],), linewidths=0.05, alpha=0.7)
    

    roa_small = np.zeros(nindex, dtype=bool)
    roa_small[ind_small] = True
    roa_small = roa_small.reshape(grid_size)
    roa_small[:,0] = False
    roa_small[:,-1] = False
    roa_small[0] = False
    roa_small[-1] = False
    ax.imshow(roa_small.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[0]), alpha=1.0)
    # ax.contour(roa_small.T, origin='lower', extent=plot_limits.ravel(), colors=(color[0],), linewidths=0.05, alpha=1.0)
    
    ax.set_xlabel(plot_labels[0], fontsize=50)
    ax.set_ylabel(plot_labels[1], fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
    ax.set_aspect(plot_limits[0][1]/plot_limits[1][1])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)
    # plt.close(fig)

def plot_levelsets(interval, margin, Tx_inv, func, plot_labels, res=32, figsize=(7,7), nlevels=10, full_path=None):
    """ take coordinate intervals and the height function and save
    the plot in full_path

    Parameter
    -------------
    interval: [[xmin, xmax], [ymin, ymax]], the limits of the axis to plot
    margin: A small value added to the plot limit
    full_path: path to save the image file
    res: resolution of the heatmap
    func: A torch (scalar-valued) function whose levelsets are to be pllotted
    nlevels = number of level sets (None for no level set)
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xmin, xmax = interval[0]
    ymin, ymax = interval[1]
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    xv, yv = np.meshgrid(x, y)
    xyv = np.vstack([xv.flatten(), yv.flatten()]).transpose()
    zv = []
    Tx_inv = torch.tensor(Tx_inv, dtype=config.ptdtype, device=config.device)
    with torch.no_grad():
        for p in xyv:
            V = torch.tensor(p, dtype=config.ptdtype, device=config.device).view(1, 2)
            zv.append(func(torch.matmul(V, Tx_inv)).detach().cpu().numpy())
    zv = np.array(zv).transpose()
    fig, ax = plt.subplots(figsize=figsize, dpi=config.dpi, frameon=False)
    ax.pcolormesh(xv, yv, zv.reshape(xv.shape), cmap='viridis')
    if nlevels is not None:
        CS = ax.contour(xv, yv, zv.reshape(xv.shape), cmap='YlOrBr', levels=nlevels)
        ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel(plot_labels[0], fontsize=50)
    ax.set_ylabel(plot_labels[1], fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if full_path is not None:
        # plt.show()
        plt.savefig(full_path, dpi=config.dpi)
    # plt.close(fig)


def plot_scalar(x_axis_vals, y_axis_vals, plot_labels, full_path=None):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
    ax.plot(x_axis_vals, y_axis_vals, linewidth=5, color=(0, 0, 0))
    ax.set_xlabel(plot_labels[0], fontsize=50)
    ax.set_ylabel(plot_labels[1], fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
    plt.xticks(np.arange(min(x_axis_vals), max(x_axis_vals)+1, 20))
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)


def plot_phase_portrait(states, closed_loop_dynamics, Tx, plot_limits, 
                        plot_labels=[None, None], full_path=None, grid_size=None,
                        ind_small=None, ind_big=None, ind_gap_stable=None, 
                        ind_exp_stable=None, ind_true_roa=None, plt_traj_dict=None):
    
    # Plot Trajectories
    if not isinstance(states, torch.Tensor):
        states_tensor = torch.tensor(states, dtype=config.ptdtype, device=config.device)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
    grad = closed_loop_dynamics(states_tensor).detach().cpu().numpy()
    grad = np.matmul(grad, Tx)
    # grad = grad / np.linalg.norm(grad, ord=2, axis=1, keepdims=True)
    states_denormalized = np.matmul(states, Tx)
    ax.quiver(states_denormalized[:,0], states_denormalized[:,1], grad[:, 0], grad[:, 1], alpha= 0.7)

    
    color=[None, None, None, None, None]
    color[0] = (80/255, 100/255, 250/255)       # ROA - bluish-green
    color[1] = (158/255, 0, 115/255)
    color[2] = (158/255, 115/255, 0)
    color[3] = (0, 158/255, 115/255)
    color[4] = (255/255, 0, 0)

    # Draw from big areas to small areas to avoid obstructing 
    nindex = grid_size[0] * grid_size[1]
    
    if ind_true_roa is not None:
        roa_true = np.zeros(nindex, dtype=bool)
        roa_true[ind_true_roa] = True
        roa_true = roa_true.reshape(grid_size)
        roa_true[:,0] = False
        roa_true[:,-1] = False
        roa_true[0] = False
        roa_true[-1] = False
        ax.imshow(roa_true.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[4]), alpha=0.4)
        # ax.contour(roa_true.T, origin='lower', extent=plot_limits.ravel(), colors=(color[4],), linewidths=0.05, alpha=0.3)

    if ind_exp_stable is not None:
        exp_stable = np.zeros(nindex, dtype=bool)
        exp_stable[ind_exp_stable] = True
        exp_stable = exp_stable.reshape(grid_size)
        exp_stable[:,0] = False
        exp_stable[:,-1] = False
        exp_stable[0] = False
        exp_stable[-1] = False
        ax.imshow(exp_stable.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[3]), alpha=0.5)
        # ax.contour(exp_stable.T, origin='lower', extent=plot_limits.ravel(), colors=(color[3],), linewidths=0.05, alpha=0.5)


    if ind_gap_stable is not None:
        roa_gap_stable = np.zeros(nindex, dtype=bool)
        roa_gap_stable[ind_gap_stable] = True
        roa_gap_stable = roa_gap_stable.reshape(grid_size)
        roa_gap_stable[:,0] = False
        roa_gap_stable[:,-1] = False
        roa_gap_stable[0] = False
        roa_gap_stable[-1] = False
        ax.imshow(roa_gap_stable.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[2]), alpha=0.6)
        # ax.contour(roa_gap_stable.T, origin='lower', extent=plot_limits.ravel(), colors=(color[2],), linewidths=0.05, alpha=0.6)

    if ind_big is not None:
        roa_big = np.zeros(nindex, dtype=bool)
        roa_big[ind_big] = True
        roa_big = roa_big.reshape(grid_size)
        roa_big[:,0] = False
        roa_big[:,-1] = False
        roa_big[0] = False
        roa_big[-1] = False
        ax.imshow(roa_big.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[1]), alpha=0.7)
        # ax.contour(roa_big.T, origin='lower', extent=plot_limits.ravel(), colors=(color[1],), linewidths=0.05, alpha=0.7)
        
    if ind_small is not None:
        roa_small = np.zeros(nindex, dtype=bool)
        roa_small[ind_small] = True
        roa_small = roa_small.reshape(grid_size)
        roa_small[:,0] = False
        roa_small[:,-1] = False
        roa_small[0] = False
        roa_small[-1] = False
        ax.imshow(roa_small.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[0]), alpha=0.8)
        # ax.contour(roa_small.T, origin='lower', extent=plot_limits.ravel(), colors=(color[0],), linewidths=0.05, alpha=1.0)

    if plt_traj_dict is not None:
        initial_states = torch.tensor(plt_traj_dict["initial_states"], dtype=config.ptdtype)
        n_trajs, sdim = initial_states.shape
        trajs = generate_trajectories(initial_states, closed_loop_dynamics, plt_traj_dict["dt"], plt_traj_dict["horizon"])
        trajs = trajs.detach().cpu().numpy()
        
        for i in range(n_trajs):
            traj = trajs[i].T
            traj = np.matmul(traj, Tx)
            ax.plot(traj[:,0], traj[:,1], '--', linewidth=4.0, alpha=1)

    ax.set_xlim(*plot_limits[0])
    ax.set_ylim(*plot_limits[1])
    ax.set_xlabel(plot_labels[0], fontsize=50)
    ax.set_ylabel(plot_labels[1], fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
    ax.set_aspect(plot_limits[0][1]/plot_limits[1][1])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)

def plot_func_levelsets(interval, margin, Tx_inv, func,  plot_labels, res=100, figsize=(10,10), nlevels=10, full_path=None):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xmin, xmax = interval[0]
    ymin, ymax = interval[1]
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    xv, yv = np.meshgrid(x, y)
    if isinstance(func, nn.Module):
        xyv = np.vstack([xv.flatten(), yv.flatten()]).transpose()
        zv = []
        Tx_inv = torch.tensor(Tx_inv, dtype=config.ptdtype, device=config.device)
        with torch.no_grad():
            for p in xyv:
                V = torch.tensor(p, dtype=config.ptdtype, device=config.device).view(1, 2)
                zv.append(func(torch.matmul(V, Tx_inv)).detach().cpu().numpy())
    else:
        zv = func(xv, yv)
    zv = np.array(zv).transpose()
    fig, ax = plt.subplots(figsize=figsize, dpi=config.dpi, frameon=False)
    pc = ax.pcolormesh(xv, yv, zv.reshape(xv.shape), cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(pc, cax=cax)    
    if nlevels is not None:
        CS = ax.contour(xv, yv, zv.reshape(xv.shape), cmap='YlOrBr', levels=nlevels)
        ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel(plot_labels[0], fontsize=50)
    ax.set_ylabel(plot_labels[1], fontsize=50)
    ax.set_xticks(np.arange(np.ceil(xmin), np.floor(xmax)+1, 1))
    ax.set_yticks(np.arange(np.ceil(ymin), np.floor(ymax)+1, 1))
    ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
    # plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.tight_layout()
    if full_path is not None:
        # plt.show()
        plt.savefig(full_path, dpi=config.dpi)
    # plt.close(fig)

def plot_traj_on_levelset(func, closed_loop_dynamics, Tx_inv, plot_limits, levels=None, 
                        plot_labels=[None, None], full_path=None, 
                        plt_traj_dict=None):
    labelsize = 50
    ticksize = 40
    linewidth = 6
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xmin, xmax = plot_limits[0]
    ymin, ymax = plot_limits[1]
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    xv, yv = np.meshgrid(x, y)
    xyv = np.vstack([xv.flatten(), yv.flatten()]).transpose()
    zv = []
    Tx = np.linalg.inv(Tx_inv)
    Tx_inv = torch.tensor(Tx_inv, dtype=config.ptdtype, device=config.device)
    with torch.no_grad():
        for p in xyv:
            V = torch.tensor(p, dtype=config.ptdtype, device=config.device).view(1, 2)
            zv.append(func(torch.matmul(V, Tx_inv)).detach().cpu().numpy())
    zv = np.array(zv).transpose()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
    pc = ax.pcolormesh(xv, yv, zv.reshape(xv.shape), cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(pc, cax=cax)
    cp = ax.contour(xv, yv, zv.reshape(xv.shape), cmap='YlOrBr', levels=levels,  linewidths = linewidth)
    ax.clabel(cp, inline=1, fontsize=10)
    

    x = np.linspace(xmin+0.1, xmax-0.1, 15)
    y = np.linspace(ymin+0.1, ymax-0.1, 15)
    xv, yv = np.meshgrid(x, y)
    states = np.vstack([xv.flatten(), yv.flatten()]).transpose()
    if not isinstance(states, torch.Tensor):
        states_tensor = torch.tensor(np.matmul(states, Tx_inv), dtype=config.ptdtype, device=config.device)
    grad = closed_loop_dynamics(states_tensor).detach().cpu().numpy()
    grad = np.matmul(grad, Tx)
    ax.quiver(states[:,0], states[:,1], grad[:, 0], grad[:, 1], alpha= 0.7)
    
    if plt_traj_dict is not None:
        initial_states = torch.tensor(plt_traj_dict["initial_states"], dtype=config.ptdtype)
        n_trajs, sdim = initial_states.shape
        trajs = generate_trajectories(initial_states, closed_loop_dynamics, plt_traj_dict["dt"], plt_traj_dict["horizon"])
        trajs = trajs.detach().cpu().numpy()
        
        for i in range(n_trajs):
            traj = trajs[i].T
            traj = np.matmul(traj, Tx)
            line = ax.plot(traj[:,0], traj[:,1], '--', linewidth=linewidth+1, alpha=1)[0]
            add_arrow(line, position=plt_traj_dict["position"], direction='right', color=None, size = None)
            ax.scatter(traj[0][0], traj[0][1], color=line.get_color(), s = 10* line.get_linewidth(), zorder=2.01) 
            
    ax.set_xlim(*plot_limits[0])
    ax.set_ylim(*plot_limits[1])
    ax.set_xlabel(plot_labels[0], fontsize=labelsize)
    ax.set_ylabel(plot_labels[1], fontsize=labelsize)
    ax.set_xticks(np.arange(np.ceil(plot_limits[0][0]), np.floor(plot_limits[0][1])+1, 1))
    ax.set_yticks(np.arange(np.ceil(plot_limits[1][0]), np.floor(plot_limits[1][1])+1, 1))
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=20)
    # ax.set_aspect(plot_limits[0][1]/plot_limits[1][1])
    ax.set_aspect('auto')
    # plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.tight_layout()
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)


def plot_scalar_function_over_trajectories(initial_states, list_of_closed_loop_dynamics, scalar_function, dt, 
                                            horizon, plot_ticks=[None, None], plot_labels=[None, None], alpha=1.0, full_path=None):
    if not isinstance(initial_states, torch.Tensor):
        initial_states = torch.tensor(initial_states, dtype=config.ptdtype, device=config.device)
    if not isinstance(list_of_closed_loop_dynamics, list):
        list_of_closed_loop_dynamics = [list_of_closed_loop_dynamics]
    n_trajs, sdim = initial_states.shape
    colors = [(158/255, 0, 115/255), (200/255, 200/255, 0)]
    p = [] * len(list_of_closed_loop_dynamics)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
    scalar_values = np.zeros((int(n_trajs), int(horizon)))
    for i, closed_loop_dynamics in enumerate(list_of_closed_loop_dynamics):    
        pt_trajectories, _ = generate_trajectories(initial_states, closed_loop_dynamics, dt, horizon)
        vals = np.zeros((n_trajs, horizon), dtype=config.dtype)
        for n in range(n_trajs):
            vals[n] = scalar_function(pt_trajectories[n].permute(1, 0)).detach().numpy().squeeze()
        x = np.arange(0, horizon, 1)
        for n in range(n_trajs):
            y = vals[n]
            ax.plot(x, y, '--', color=colors[i], linewidth=2.0, alpha=alpha)
    
    lines = [Line2D([0], [0], color=c, linewidth=3.0, linestyle='--', alpha=alpha) for c in colors]    
    labels = ['Untrained Controller', 'Trained Controller']
    legend = ax.legend(lines, labels, fontsize=20)
    legend.get_frame().set_alpha(0.5)

    ax.set_facecolor((0.0, 76/255, 153/255))
    ax.set_xlabel(plot_labels[0])
    ax.set_ylabel(plot_labels[1])
    ax.set_xlabel(plot_labels[0], fontsize=20)
    ax.set_ylabel(plot_labels[1], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15, grid_linewidth=15)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)

    
def plot_levelsets_as_binary_maps(gridsize, plot_limits, lyapunov, c_values, 
                    plot_labels=[None, None], full_path=None):

    """ Take a function and plot its levelsets as binary maps

    Parameter
    -------------
    gridsize : Tuple, Size of the rectangular grid on which the levelsets are plotted
    lyapunov: A Lyapunov class instance, It contains its values as a property
    c_values: Iterable, The values of the function corresponding to the requested levelsets
    plot_labels: [label of x axis, label of y axis]
    
    """

    dir_path = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    color=[None]
    color[0] = (0, 158/255, 115/255)       # ROA - bluish-green
    nrows, ncolumns = get_number_of_rows_and_columns(len(c_values))
    for i, c in enumerate(c_values):
        z = (lyapunov.values.detach().numpy() < c).reshape(gridsize)
        alpha = 1
        ax = plt.subplot(nrows, ncolumns, i+1)
        ax.set_title("c={:10.5}".format(c))
        ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(color[0],), linewidths=1)
        ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[0]), alpha=alpha)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=20)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        if full_path is not None:
            plt.savefig(full_path, dpi=config.dpi)


def plot_2d_streamplot(interval, df, density=1, margin=10, res=30, plot_labels=[None, None], plot_ticks=[None, None], full_path=None):
    """
    Takes the 2D df (rhs of ODE) and plots its stream plot in the specified interval.
    interval : [[xmin, xmax], [ymin, ymax]]
    df: takes (x, y) and outputs (xdot, ydot)
    density:density of the streamplot
    margin: margin of the plot in addition to the xmin/max and ymin/ymax
    res: determines the density of the flow
    alpha: opacity of the flow plot
    full_path: the path to solve the plot
    """
    
    # plot levelsets
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    colors = [(200/255, 200/255, 0), (158/255, 0, 115/255)]
    xmin, xmax = interval[0]
    ymin, ymax = interval[1]
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin
    X, Y = np.meshgrid(np.linspace(xmin, xmax, res), np.linspace(ymin, ymax, res))
    u, v = np.zeros_like(X), np.zeros_like(Y)
    NI, NJ = X.shape
    for i in range(NI):
        for j in range(NJ):
            x, y = X[i, j], Y[i, j]
            dx, dy = df(x, y)
            u[i,j] = dx
            v[i,j] = dy
            
    fig, ax = plt.subplots(figsize=(10,10), dpi=config.dpi, frameon=False)
    ax.streamplot(X, Y, u, v, density=density)
    ax.set_xlabel(plot_labels[0], fontsize=30)
    ax.set_ylabel(plot_labels[1], fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20, grid_linewidth=20)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.axis('square')
    plt.axis([xmin, xmax, ymin, ymax])
    ax.xaxis.set_ticks(plot_ticks[0])
    ax.yaxis.set_ticks(plot_ticks[1])
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)


def plot_roa_3D(points, plot_limits, plot_labels, full_path):
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    ax.scatter(x, y, z)
    ax.set_xlabel(plot_labels[0])
    ax.set_ylabel(plot_labels[1])
    ax.set_zlabel(plot_labels[2])
    ax.set_xlim(plot_limits[0])
    ax.set_ylim(plot_limits[1])
    ax.set_zlim(plot_limits[2])
    ax.view_init(elev=20, azim=-70) 

    plt.tight_layout()
    plt.savefig(full_path, dpi=config.dpi)

def plot_roa_3D_old(roa, Tx, plot_labels, full_path):
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    x, y, z = np.indices(np.array(roa.shape)+1)
    x = (x/np.max(x)*2 - 1)*Tx[0,0]
    y = (y/np.max(y)*2 - 1)*Tx[1,1]
    z = (z/np.max(z)*2 - 1)*Tx[2,2]

    ax.voxels(x, y, z, roa, facecolors=(255/255, 0, 0, 0.5))
    ax.voxels(x, y, z, roa, facecolors=(80/255, 100/255, 250/255, 0.5))
    ax.set_xlabel(plot_labels[0])
    ax.set_ylabel(plot_labels[1])
    ax.set_zlabel(plot_labels[2])
    # ax.set_xlim(plot_limits[0])
    # ax.set_ylim(plot_limits[1])
    # ax.set_zlim(plot_limits[2])
    ax.view_init(elev=20, azim=-70) 

    plt.tight_layout()
    plt.savefig(full_path, dpi=config.dpi)

def plot_nested_roas_3D(grid, Tx, ind_small, ind_big, plot_limits, plot_labels, ind_gap_stable=None, ind_exp_stable=None, ind_true_roa=None, full_path=None):
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    color=[None, None, None, None, None]
    color[0] = (80/255, 100/255, 250/255, 1.0)     
    color[1] = (158/255, 0, 115/255, 0.5)
    color[2] = (158/255, 115/255, 0, 0.3)
    color[3] = (0, 158/255, 115/255, 0.2)
    color[4] = (255/255, 0, 0, 0.1)

    # Draw from big areas to small areas to avoid obstructing
    grid_size = grid.num_points
    x, y, z = np.indices(np.array(grid_size)+1)
    x = (x/np.max(x)*2 - 1)*Tx[0,0]
    y = (y/np.max(y)*2 - 1)*Tx[1,1]
    z = (z/np.max(z)*2 - 1)*Tx[2,2]

    tmp = np.matmul(grid.all_points, Tx)
    xs, ys, zs = tmp[:,0], tmp[:,1], tmp[:,2]
    if ind_true_roa is not None:
        roa_true = ind_true_roa.reshape(grid_size)
        # ax.voxels(x, y, z, roa_true, facecolors=color[4])
        ax.scatter(xs[ind_true_roa], ys[ind_true_roa], zs[ind_true_roa], color = color[4])

  
    if ind_exp_stable is not None:
        exp_stable = ind_exp_stable.reshape(grid_size)
        # ax.voxels(x, y, z, exp_stable, facecolors=color[3])
        ax.scatter(xs[ind_exp_stable], ys[ind_exp_stable], zs[ind_exp_stable], color = color[3])

    if ind_gap_stable is not None:
        roa_gap_stable = ind_gap_stable.reshape(grid_size)
        # ax.voxels(x, y, z, roa_gap_stable, facecolors=color[2])
        ax.scatter(xs[ind_gap_stable], ys[ind_gap_stable], zs[ind_gap_stable], color = color[2])
   
    roa_big = ind_big.reshape(grid_size)
    # ax.voxels(x, y, z, roa_big, facecolors=color[1])
    ax.scatter(xs[ind_big], ys[ind_big], zs[ind_big], color = color[1])

    roa_small = ind_small.reshape(grid_size)
    # ax.voxels(x, y, z, roa_small, facecolors=color[0])
    ax.scatter(xs[ind_small], ys[ind_small], zs[ind_small], color = color[0])

    ax.set_xlabel(plot_labels[0])
    ax.set_ylabel(plot_labels[1])
    ax.set_zlabel(plot_labels[2])
    ax.set_xlim(plot_limits[0])
    ax.set_ylim(plot_limits[1])
    ax.set_zlim(plot_limits[2])
    ax.view_init(elev=20, azim=-70) 

    plt.tight_layout()
    plt.savefig(full_path, dpi=config.dpi)
    # plt.savefig(full_path, dpi=50)

def plot_nested_roas_3D_diagnostic(grid, Tx, ind_small, ind_big, plot_limits, plot_labels, ind_gap_stable=None, ind_exp_stable=None, ind_true_roa=None, full_path=None):
    labelsize = 50
    ticksize = 40
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    color=[None, None, None, None, None]
    cmap=plt.get_cmap('Paired')
    color[0] = (*cmap(0.1)[0:3], 1.0) # ind_small
    color[1] = (*cmap(0.6)[0:3], 0.5) # ind_big
    color[2] = (*cmap(0.5)[0:3], 0.3) # not quite used
    color[3] = (*cmap(0.45)[0:3], 0.2) # ind_exp_stable
    color[4] = (*cmap(0.3)[0:3], 0.2) # all_roa

    # Draw from big areas to small areas to avoid obstructing
    grid_size = grid.num_points
    x, y, z = np.indices(np.array(grid_size)+1)
    x = (x/np.max(x)*2 - 1)*Tx[0,0]
    y = (y/np.max(y)*2 - 1)*Tx[1,1]
    z = (z/np.max(z)*2 - 1)*Tx[2,2]

    tmp = np.matmul(grid.all_points, Tx)
    xs, ys, zs = tmp[:,0], tmp[:,1], tmp[:,2]
    if ind_true_roa is not None:
        roa_true = ind_true_roa.reshape(grid_size)
        # ax.voxels(x, y, z, roa_true, facecolors=color[4])
        ax.scatter(xs[ind_true_roa], ys[ind_true_roa], zs[ind_true_roa], color = color[4])

  
    if ind_exp_stable is not None:
        exp_stable = ind_exp_stable.reshape(grid_size)
        # ax.voxels(x, y, z, exp_stable, facecolors=color[3])
        ax.scatter(xs[ind_exp_stable], ys[ind_exp_stable], zs[ind_exp_stable], color = color[3])

    if ind_gap_stable is not None:
        roa_gap_stable = ind_gap_stable.reshape(grid_size)
        # ax.voxels(x, y, z, roa_gap_stable, facecolors=color[2])
        ax.scatter(xs[ind_gap_stable], ys[ind_gap_stable], zs[ind_gap_stable], color = color[2])
    
    if ind_big is not None:
        roa_big = ind_big.reshape(grid_size)
        # ax.voxels(x, y, z, roa_big, facecolors=color[1])
        ax.scatter(xs[ind_big], ys[ind_big], zs[ind_big], color = color[1])

    if ind_small is not None:
        roa_small = ind_small.reshape(grid_size)
        # ax.voxels(x, y, z, roa_small, facecolors=color[0])
        ax.scatter(xs[ind_small], ys[ind_small], zs[ind_small], color = color[0])

    ax.set_xlabel(plot_labels[0], fontsize=labelsize)
    ax.set_ylabel(plot_labels[1], fontsize=labelsize)
    ax.set_zlabel(plot_labels[2], fontsize=labelsize)
    ax.xaxis.labelpad=20
    ax.yaxis.labelpad=20
    ax.zaxis.labelpad=5
    # ax.dist = 13
    ax.set_xlim(plot_limits[0])
    ax.set_ylim(plot_limits[1])
    ax.set_zlim(plot_limits[2])
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=20)
    ax.view_init(elev=20, azim=-70) 

    plt.tight_layout()
    plt.savefig(full_path, dpi=config.dpi)
    # plt.savefig(full_path, dpi=50)

def plot_nested_roas(grid_size, ind_small, ind_big, plot_limits, plot_labels, ind_gap_stable=None, ind_exp_stable=None, ind_true_roa=None, full_path=None):
    """ take a grid and the indices for small and big ROAs and plot them overlaid
    the plot in full_path

    Parameter
    -------------
    grid_size: size of the encompassing grid for both ROAs
    ind_small: binary ndarray vector, indices for the points of the inner ROA
    ind_big: binary ndarray vector, indices for points of the outer ROA
    ind_points, ind_gap_stable, ind_true_roa: optional, a binary vector with the same size of prod(grid_size).
            Each group of points are plotted with different colour to differentiate betweent the nature of points.
    plot_limits: [(xmin, xmax), (ymin, ymax)]
    full_path: path to save the image file
    """
    if full_path is not None: 
        dir_path = os.path.dirname(full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(111)
    color=[None, None, None, None, None]
    color[0] = (80/255, 100/255, 250/255)       # ROA - bluish-green
    color[1] = (158/255, 0, 115/255)
    color[2] = (158/255, 115/255, 0)
    color[3] = (0, 158/255, 115/255)
    color[4] = (255/255, 0, 0)

    # Draw from big areas to small areas to avoid obstructing 
    nindex = grid_size[0] * grid_size[1]
    
    if ind_true_roa is not None:
        roa_true = np.zeros(nindex, dtype=bool)
        roa_true[ind_true_roa] = True
        roa_true = roa_true.reshape(grid_size)
        roa_true[:,0] = False
        roa_true[:,-1] = False
        roa_true[0] = False
        roa_true[-1] = False
        ax.imshow(roa_true.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[4]), alpha=0.4)
        # ax.contour(roa_true.T, origin='lower', extent=plot_limits.ravel(), colors=(color[4],), linewidths=0.05, alpha=0.3)

    if ind_exp_stable is not None:
        exp_stable = np.zeros(nindex, dtype=bool)
        exp_stable[ind_exp_stable] = True
        exp_stable = exp_stable.reshape(grid_size)
        exp_stable[:,0] = False
        exp_stable[:,-1] = False
        exp_stable[0] = False
        exp_stable[-1] = False
        ax.imshow(exp_stable.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[3]), alpha=0.5)
        # ax.contour(exp_stable.T, origin='lower', extent=plot_limits.ravel(), colors=(color[3],), linewidths=0.05, alpha=0.5)


    if ind_gap_stable is not None:
        roa_gap_stable = np.zeros(nindex, dtype=bool)
        roa_gap_stable[ind_gap_stable] = True
        roa_gap_stable = roa_gap_stable.reshape(grid_size)
        roa_gap_stable[:,0] = False
        roa_gap_stable[:,-1] = False
        roa_gap_stable[0] = False
        roa_gap_stable[-1] = False
        ax.imshow(roa_gap_stable.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[2]), alpha=0.6)
        # ax.contour(roa_gap_stable.T, origin='lower', extent=plot_limits.ravel(), colors=(color[2],), linewidths=0.05, alpha=0.6)

    if ind_big is not None:
        roa_big = np.zeros(nindex, dtype=bool)
        roa_big[ind_big] = True
        roa_big = roa_big.reshape(grid_size)
        roa_big[:,0] = False
        roa_big[:,-1] = False
        roa_big[0] = False
        roa_big[-1] = False
        ax.imshow(roa_big.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[1]), alpha=0.7)
        # ax.contour(roa_big.T, origin='lower', extent=plot_limits.ravel(), colors=(color[1],), linewidths=0.05, alpha=0.7)
        

    roa_small = np.zeros(nindex, dtype=bool)
    roa_small[ind_small] = True
    roa_small = roa_small.reshape(grid_size)
    roa_small[:,0] = False
    roa_small[:,-1] = False
    roa_small[0] = False
    roa_small[-1] = False
    ax.imshow(roa_small.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[0]), alpha=1.0)
    # ax.contour(roa_small.T, origin='lower', extent=plot_limits.ravel(), colors=(color[0],), linewidths=0.05, alpha=1.0)
    
    ax.set_xlabel(plot_labels[0], fontsize=50)
    ax.set_ylabel(plot_labels[1], fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
    ax.set_aspect(plot_limits[0][1]/plot_limits[1][1])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)
    # plt.close(fig)

def plot_nested_roas_diagnostic(grid, Tx, ind_small, ind_big, plot_limits, plot_labels, ind_gap_stable=None, ind_exp_stable=None, ind_true_roa=None, full_path=None, add_lengend = False):
    labelsize = 70 # 50
    ticksize = 60 # 40
    legendsize = 41 # 30
    markersize = 20
    # marker_linewidth = 
    division = 13
    grid_size = grid.num_points
    if full_path is not None: 
        dir_path = os.path.dirname(full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(111)
    color=[None, None, None, None, None]
    backgroud = (1, 1, 1, 1)
    # cmap=plt.get_cmap('Paired')
    # color[0] = cmap(0.1)[0:4] # ind_small
    # color[1] = cmap(0.0)[0:4] # ind_big
    # color[2] = cmap(0.4)[0:4] # not quite used
    # color[3] = cmap(0.2)[0:4] # forward invariant roa
    # color[4] = cmap(0.3)[0:4] # all_roa
    
    cmap=plt.get_cmap('tab20c')
    color[0] = cmap(0.0)[0:4] # ind_small
    color[1] = cmap(0.05)[0:4] # ind_big
    color[2] = cmap(0.4)[0:4] # not quite used
    color[3] = cmap(0.1)[0:4] # forward invariant roa
    color[4] = cmap(0.15)[0:4] # all_roa

    # cmap=plt.get_cmap('tab20c')
    # color[0] = cmap(0.0)[0:4] # ind_small
    # color[1] = cmap(0.05)[0:4] # ind_big
    # color[2] = cmap(0.5)[0:4] # not quite used
    # color[3] = cmap(0.5)[0:4] # forward invariant roa
    # color[4] = cmap(0.55)[0:4] # all_roa

    nindex = grid_size[0] * grid_size[1]
    legend_colors = []
    legend_names = []
    overall_matrix = np.zeros(nindex)
    if ind_true_roa is not None:
        roa_true = np.zeros(nindex, dtype=bool)
        roa_true[ind_true_roa] = True
        roa_true = roa_true.reshape(grid_size)
        roa_true[:,0] = False
        roa_true[:,-1] = False
        roa_true[0] = False
        roa_true[-1] = False
        roa_true = roa_true.reshape(nindex)
        overall_matrix[roa_true] = 1
        legend_colors.append(color[4])

    if ind_exp_stable is not None:
        exp_stable = np.zeros(nindex, dtype=bool)
        exp_stable[ind_exp_stable] = True
        exp_stable = exp_stable.reshape(grid_size)
        exp_stable[:,0] = False
        exp_stable[:,-1] = False
        exp_stable[0] = False
        exp_stable[-1] = False
        exp_stable = exp_stable.reshape(nindex)
        overall_matrix[exp_stable] = 2
        legend_colors.append(color[3])

    if ind_gap_stable is not None:
        roa_gap_stable = np.zeros(nindex, dtype=bool)
        roa_gap_stable[ind_gap_stable] = True
        roa_gap_stable = roa_gap_stable.reshape(grid_size)
        roa_gap_stable[:,0] = False
        roa_gap_stable[:,-1] = False
        roa_gap_stable[0] = False
        roa_gap_stable[-1] = False

    if ind_big is not None:
        roa_big = np.zeros(nindex, dtype=bool)
        roa_big[ind_big] = True
        roa_big = roa_big.reshape(grid_size)
        roa_big[:,0] = False
        roa_big[:,-1] = False
        roa_big[0] = False
        roa_big[-1] = False
        roa_big = roa_big.reshape(nindex)
        overall_matrix[roa_big] = 3
        legend_colors.append(color[1])

    if ind_small is not None:
        roa_small = np.zeros(nindex, dtype=bool)
        roa_small[ind_small] = True
        roa_small = roa_small.reshape(grid_size)
        roa_small[:,0] = False
        roa_small[:,-1] = False
        roa_small[0] = False
        roa_small[-1] = False
        roa_small = roa_small.reshape(nindex)
        overall_matrix[roa_small] = 4
        legend_colors.append(color[0])

    cmap = [backgroud] + legend_colors
    overall_matrix = overall_matrix.reshape(grid_size)
    ax.imshow(overall_matrix.T, origin='lower', extent=plot_limits.ravel(), cmap=ListedColormap(cmap), alpha=1.0)
    ax.set_xlabel(plot_labels[0], fontsize=labelsize)
    ax.set_ylabel(plot_labels[1], fontsize=labelsize)
    ax.set_xticks(np.arange(np.ceil(plot_limits[0][0]), np.floor(plot_limits[0][1])+1, 1))
    ax.set_yticks(np.arange(np.ceil(plot_limits[1][0]), np.floor(plot_limits[1][1])+1, 1))
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=20)
    ax.set_aspect('auto')
    if add_lengend == True:
        names = ['RoA', 'Forward Invariant RoA', 'Sampling Area', 'Estimated RoA']
        patches = [mpatches.Patch(color=legend_colors[i], label=names[i]) for i in range(len(names))]
        # ax.legend(handles=patches, loc='best', borderaxespad=0., fontsize = legendsize)
        ax.legend(handles=patches, bbox_to_anchor=(0.99, 0.99), loc="upper right", borderaxespad=0. , fontsize = legendsize)
    plt.tight_layout()
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)
    # plt.close(fig)


def plot_nested_roas_diagnostic_old(grid, Tx, ind_small, ind_big, plot_limits, plot_labels, ind_gap_stable=None, ind_exp_stable=None, ind_true_roa=None, full_path=None):
    labelsize = 50
    ticksize = 40
    markersize = 20
    # marker_linewidth = 
    division = 13
    grid_size = grid.num_points
    if full_path is not None: 
        dir_path = os.path.dirname(full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(111)
    color=[None, None, None, None, None]
    cmap=plt.get_cmap('Paired')
    color[0] = cmap(0.1)[0:3] # ind_small
    color[1] = cmap(0.0)[0:3] # ind_big
    color[2] = cmap(0.4)[0:3] # not quite used
    color[3] = cmap(0.2)[0:3] # forward invariant roa
    color[4] = cmap(0.3)[0:3] # all_roa

    # Draw from big areas to small areas to avoid obstructing 
    nindex = grid_size[0] * grid_size[1]
    
    if ind_true_roa is not None:
        roa_true = np.zeros(nindex, dtype=bool)
        roa_true[ind_true_roa] = True
        roa_true = roa_true.reshape(grid_size)
        roa_true[:,0] = False
        roa_true[:,-1] = False
        roa_true[0] = False
        roa_true[-1] = False
        ax.imshow(roa_true.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[4]), alpha=1, zorder = 2.1)
        roa_true = roa_true.reshape(nindex)
        roa_true[ind_exp_stable] = False
        # roa_true[::step] = False
        pt = np.matmul(grid.all_points[roa_true], Tx)
        # ind = np.where(roa_true==True)[0]
        # batch_inds = np.random.choice(ind.shape[0], ind.shape[0]//division, replace=False)
        # pt = np.matmul(grid.all_points[ind[batch_inds]], Tx)
        ax.scatter(pt[:,0], pt[:,1], s = markersize, c = 'green', marker='x', zorder = 2.1)
        # ax.contour(roa_true.T, origin='lower', extent=plot_limits.ravel(), colors=(color[4],), linewidths=0.05, alpha=0.3)

    if ind_exp_stable is not None:
        exp_stable = np.zeros(nindex, dtype=bool)
        exp_stable[ind_exp_stable] = True
        exp_stable = exp_stable.reshape(grid_size)
        exp_stable[:,0] = False
        exp_stable[:,-1] = False
        exp_stable[0] = False
        exp_stable[-1] = False
        ax.imshow(exp_stable.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[3]), alpha=1, zorder = 2.2)
        # pt = np.matmul(grid.all_points[ind_exp_stable], Tx)
        # ax.scatter(pt[:,0], pt[:,1], s = 7, c = 'darkgreen', marker='x', zorder = 2.2)
        # ax.contour(exp_stable.T, origin='lower', extent=plot_limits.ravel(), colors=(color[3],), linewidths=0.05, alpha=0.5)


    if ind_gap_stable is not None:
        roa_gap_stable = np.zeros(nindex, dtype=bool)
        roa_gap_stable[ind_gap_stable] = True
        roa_gap_stable = roa_gap_stable.reshape(grid_size)
        roa_gap_stable[:,0] = False
        roa_gap_stable[:,-1] = False
        roa_gap_stable[0] = False
        roa_gap_stable[-1] = False
        ax.imshow(roa_gap_stable.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[2]), alpha=1, zorder = 2.3)
        # ax.contour(roa_gap_stable.T, origin='lower', extent=plot_limits.ravel(), colors=(color[2],), linewidths=0.05, alpha=0.6)

    if ind_big is not None:
        roa_big = np.zeros(nindex, dtype=bool)
        roa_big[ind_big] = True
        roa_big = roa_big.reshape(grid_size)
        roa_big[:,0] = False
        roa_big[:,-1] = False
        roa_big[0] = False
        roa_big[-1] = False
        ax.imshow(roa_big.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[1]), alpha=1, zorder = 2.4)
        roa_big = roa_big.reshape(nindex)
        roa_big[ind_small] = False
        # roa_big[::step] = False
        pt = np.matmul(grid.all_points[roa_big], Tx)
        # ind = np.where(roa_big==True)[0]
        # batch_inds = np.random.choice(ind.shape[0], ind.shape[0]//division, replace=False)
        # pt = np.matmul(grid.all_points[ind[batch_inds]], Tx)
        ax.scatter(pt[:,0], pt[:,1], s = markersize, c = 'steelblue', marker='+', zorder = 2.4)
        # ax.contour(roa_big.T, origin='lower', extent=plot_limits.ravel(), colors=(color[1],), linewidths=0.05, alpha=0.7)
        
    if ind_small is not None:
        roa_small = np.zeros(nindex, dtype=bool)
        roa_small[ind_small] = True
        roa_small = roa_small.reshape(grid_size)
        roa_small[:,0] = False
        roa_small[:,-1] = False
        roa_small[0] = False
        roa_small[-1] = False
        ax.imshow(roa_small.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[0]), alpha=1.0, zorder = 2.5)
        # ax.contour(roa_small.T, origin='lower', extent=plot_limits.ravel(), colors=(color[0],), linewidths=0.05, alpha=1.0)
    
    ax.set_xlabel(plot_labels[0], fontsize=labelsize)
    ax.set_ylabel(plot_labels[1], fontsize=labelsize)
    ax.set_xticks(np.arange(np.ceil(plot_limits[0][0]), np.floor(plot_limits[0][1])+1, 1))
    ax.set_yticks(np.arange(np.ceil(plot_limits[1][0]), np.floor(plot_limits[1][1])+1, 1))
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=20)
    # ax.set_aspect(plot_limits[0][1]/plot_limits[1][1])
    ax.set_aspect('auto')
    # plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.tight_layout()
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)
    # plt.close(fig)
