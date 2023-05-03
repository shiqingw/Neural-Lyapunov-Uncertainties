import numpy as np
import torch
from .utils import batchify, get_storage, set_storage, unique_rows
from .configuration import Configuration
config = Configuration()
del Configuration


__all__ = ['Lyapunov_CT']

class Lyapunov_CT(object):
    """A class for general Lyapunov functions.

    Parameters
    ----------
    discretization : ndarray
        A discrete grid on which to evaluate the Lyapunov function.
    lyapunov_function : callable or instance of `DeterministicFunction`
        The lyapunov function. Can be called with states and returns the
        corresponding values of the Lyapunov function.
    dynamics : a callable or an instance of `Function`
        The dynamics model. Can be either a deterministic function or something
        uncertain that includes error bounds.
    lipschitz_dynamics : ndarray or float
        The Lipschitz constant of the dynamics. Either globally, or locally
        for each point in the discretization (within a radius given by the
        discretization constant. This is the closed-loop Lipschitz constant
        including the policy!
    lipschitz_lyapunov : ndarray or float
        The Lipschitz constant of the lyapunov function. Either globally, or
        locally for each point in the discretization (within a radius given by
        the discretization constant.
    tau : float
        The discretization constant.
    policy : ndarray, optional
        The control policy used at each state (Same number of rows as the
        discretization).
    initial_set : ndarray, optional
        A boolean array of states that are known to be safe a priori.
    adaptive : bool, optional
        A boolean determining whether an adaptive discretization is used for
        stability verification.
    decrease_thresh: None or a real value. If None, the threshold is computed by self.threshold function.
    If it is a real value, the value is considered as the threshold.

    """

    def __init__(self, discretization, lyapunov_function, grad_lyapunov_function, closed_loop_dynamics_true, 
                closed_loop_dynamics_nominal, lipschitz_dynamics, lipschitz_lyapunov, lipschitz_gradient_lyapunov,
                tau, initial_set=None, adaptive=False, decrease_thresh=0):
        """Initialization, see `Lyapunov` for details."""
        super(Lyapunov_CT, self).__init__()

        self.discretization = discretization
        # Keep track of the safe sets
        self.safe_set_true = np.zeros(np.prod(discretization.num_points), dtype=bool)
        self.largest_safe_set_true = np.zeros(np.prod(discretization.num_points), dtype=bool)
        self.exp_stable_set_true = np.zeros(np.prod(discretization.num_points), dtype=bool)
        self.largest_exp_stable_set_true = np.zeros(np.prod(discretization.num_points), dtype=bool)

        self.safe_set_nominal = np.zeros(np.prod(discretization.num_points), dtype=bool)
        self.largest_safe_set_nominal = np.zeros(np.prod(discretization.num_points), dtype=bool)
        self.exp_stable_set_nominal = np.zeros(np.prod(discretization.num_points), dtype=bool)
        self.largest_exp_stable_set_nominal = np.zeros(np.prod(discretization.num_points), dtype=bool)

        self.c_max_true = 0
        self.c_max_exp_true = 0
        self.c_max_nominal = 0
        self.c_max_exp_nominal = 0

        self.initial_safe_set = initial_set
        if initial_set is not None:
            self.safe_set_true[initial_set] = True
            self.largest_safe_set_true[initial_set] = True
            self.exp_stable_set_true[initial_set] = True
            self.largest_exp_stable_set_true[initial_set] = True
            self.safe_set_nominal[initial_set] = True
            self.largest_safe_set_nominal[initial_set] = True
            self.exp_stable_set_nominal[initial_set] = True
            self.largest_exp_stable_set_nominal[initial_set] = True

        # Discretization constant
        self.tau = tau
        self.decrease_thresh = decrease_thresh
        # Make sure dynamics are of standard framework
        self.closed_loop_dynamics_true = closed_loop_dynamics_true
        self.closed_loop_dynamics_nominal = closed_loop_dynamics_nominal
        # Make sure Lyapunov fits into standard framework
        self.lyapunov_function = lyapunov_function
        self.grad_lyapunov_function = grad_lyapunov_function
        # Storage for graph
        self._storage = dict()
        # Lyapunov values
        self.values = None
        self._lipschitz_dynamics = lipschitz_dynamics
        self._lipschitz_lyapunov = lipschitz_lyapunov
        self.lipschitz_gradient_lyapunov = lipschitz_gradient_lyapunov
        self.update_values()
        self.adaptive = adaptive


    def lipschitz_dynamics(self, states):
        """Return the Lipschitz constant for given states and actions.

        Parameters
        ----------
        states : ndarray or Tensor

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            If lipschitz_dynamics is a callable then returns local Lipschitz
            constants. Otherwise returns the Lipschitz constant as a scalar.
        """
        if hasattr(self._lipschitz_dynamics, '__call__'): # check if _lipschitz_dynamics is a function
            return self._lipschitz_dynamics(states)
        else:
            return self._lipschitz_dynamics


    def lipschitz_lyapunov(self, states):
        """Return the local Lipschitz constant at a given state.

        Parameters
        ----------
        states : ndarray or Tensor

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            If lipschitz_lyapunov is a callable then returns local Lipschitz
            constants. Otherwise returns the Lipschitz constant as a scalar.
        """
        if hasattr(self._lipschitz_lyapunov, '__call__'):
            return self._lipschitz_lyapunov(states)
        else:
            return self._lipschitz_lyapunov


    def threshold(self, states, tau=None):
        """Return the safety threshold for the Lyapunov condition.
        meaning that V_dot must be less than this threshold
        to ensure negativity of the V_dot

        Parameters
        ----------
        states : ndarray or torch.Tensor

        tau : np.float or torch.Tensor, optional
            discretization constant to consider.

        Returns
        -------
        lipschitz : np.float, ndarray or torch.Tensor
            Either the scalar threshold or local thresholds, depending on
            whether lipschitz_lyapunov and lipschitz_dynamics are local or not.
        """
        if tau is None:
            tau = self.tau
        Lv = self.lipschitz_lyapunov(states)
        if hasattr(self._lipschitz_lyapunov, '__call__') and Lv.shape[1] > 1:
            Lv = torch.norm(Lv, p=1, axis=1)
        Lf = self.lipschitz_dynamics(states)
        LdV = self.lipschitz_gradient_lyapunov(states)
        Bf = torch.norm(self.closed_loop_dynamics_true(states), p=1, dim=1, keepdim=True)
        L =  torch.squeeze(Lv*Lf + Bf*LdV)
        return - L * tau


    def is_safe(self, state):
        """Return a boolean array that indicates whether the state is safe using the current safe set.

        Parameters
        ----------
        state : ndarray

        Returns
        -------
        safe : boolean numpy array
            Is true if the corresponding state is inside the safe set.

        """
        return self.safe_set_true[self.discretization.state_to_index(state)]


    def update_values(self):
        """Update the discretized values when the Lyapunov function changes.
        self.values will be a 1D torch Tensor, (N, ) tensor of scalars where N is the number of
        points in the discretization.
        It also updates the self._storage
        """
    
        storage = get_storage(self._storage)
        if storage is None:
            pt_points = self.discretization.all_points
            pt_values = self.lyapunov_function(pt_points)
            storage = [('points', pt_points), ('values', pt_values)]
            set_storage(self._storage, storage)
        else:
            pt_points, pt_values = storage.values()
        pt_points = self.discretization.all_points
        with torch.no_grad(): 
            self.values = torch.squeeze(self.lyapunov_function(pt_points))

    def check_decrease_condition_v_dot(self, pt_states, closed_loop_dynamics, threshold):
        """ Check if the decrease condition is satisfied for the points on the dicretization for a given policy

        Parameters
        ----------
        pt_states: (N x d) pytorch tensors as the states of the system
        policy: A pytorch function that determines how actions are produced by the current states. If policy is None, the system
                is autonomous.
        threshold: (N x 1) negative values as the upper bound of the decrease condition of the Lyapunov function for each state

        Returns
        ----------
        decrease_condition: (N,) pytorch tensor representing if the decrease condition holds for each state
        """
        if not isinstance(pt_states, torch.Tensor):
            pt_states = torch.tensor(pt_states, dtype=config.ptdtype, device=config.device)
        
        dynamics =  closed_loop_dynamics(pt_states)
        decrease = torch.sum(torch.mul(self.grad_lyapunov_function(pt_states), dynamics),1)
        pt_negative = torch.squeeze(torch.lt(decrease, threshold))
        return pt_negative

    def check_decrease_condition(self, pt_states, closed_loop_dynamics, dt, horizon, tol):
        """ Check if the decrease condition is satisfied for the points on the dicretization for a given policy

        Parameters
        ----------
        pt_states: (N x d) np.array as the states of the system
        policy: A pytorch function that determines how actions are produced by the current states. If policy is None, the system
                is autonomous.
        threshold: (N x 1) negative values as the upper bound of the decrease condition of the Lyapunov function for each state

        Returns
        ----------
        decrease_condition: (N,) pytorch tensor representing if the decrease condition holds for each state
        """

        end_states = pt_states

        with torch.no_grad():
            for t in range(1, horizon):
                end_states = closed_loop_dynamics(end_states).detach().cpu().numpy()*dt + end_states
        
        equilibrium = np.zeros((1, pt_states.shape[1]))

        # Compute an approximate ROA as all states that end up "close" to 0
        dists = np.linalg.norm(end_states - equilibrium, ord=2, axis=1, keepdims=False)
        stable = (dists <= tol)

        return stable


    def update_safe_set(self, true_or_nominal, roa_true):
        if true_or_nominal != "true" and true_or_nominal != "nominal":
            raise ValueError("Have to choose between true or nominal!")
        closed_loop_dynamics = getattr(self, 'closed_loop_dynamics_' + true_or_nominal)
        
        # Reset the safe set
        safe_set = np.zeros(np.prod(self.discretization.num_points), dtype=bool)
        if self.initial_safe_set is not None:
            safe_set[self.initial_safe_set] = True
        # # Assume cannot shrink
        # safe_set = np.logical_or(safe_set, getattr(self, 'safe_set_' + true_or_nominal))

        self.update_values()
        value_order = np.argsort(np.squeeze(self.values.detach().cpu().numpy())) # ordered indices based on the values of the Lyapunov function
        safe_set = safe_set[value_order]
        roa_true_ordered = roa_true[value_order]

        # Verify safety in batches
        batch_size = config.batch_size
        batch_generator = batchify((value_order, safe_set, roa_true_ordered),
                                   batch_size)
        index_to_state = self.discretization.index_to_state

        #######################################################################

        for i, (indices, safe_batch, roa_true_batch) in batch_generator:
            states = index_to_state(indices)
            # Update the safety with the safe_batch result
            thresh = torch.tensor(self.decrease_thresh, dtype=config.ptdtype, device=config.device) \
                if self.decrease_thresh is not None \
                else self.threshold(torch.tensor(states, dtype=config.ptdtype), self.tau)
            
            stable = self.check_decrease_condition_v_dot(states, closed_loop_dynamics, thresh).detach().cpu().numpy()
            safe_batch |= stable
            safe_batch &= roa_true_batch

            # Boolean array: argmin returns first element that is False
            # If all are safe then it returns 0
            bound = np.argmin(safe_batch)

            # Check if there are unsafe elements in the batch
            if bound > 0 or not safe_batch[0]:
                # Make sure all following points are labeled as unsafe (because the batch is ordered with the values of the Lyapunov function)
                safe_batch[bound:] = False
                break
        # The largest index of a safe value
        max_index = max(i + bound - 1, 0) # i is the starting index of each batch and bound is the index inside a batch
        # Set placeholder for c_max to the corresponding value
        c_max_tmp = self.values[value_order[max_index]].detach().cpu().numpy().item()
        setattr(self, 'c_max_unconstrained_' + true_or_nominal, c_max_tmp)
        # Restore the order of the safe set
        safe_nodes = value_order[safe_set]
        safe_set[:] = False
        safe_set[safe_nodes] = True

        # Ensure the initial safe set is kept
        if self.initial_safe_set is not None:
            safe_set[self.initial_safe_set] = True

        setattr(self, 'safe_set_' + true_or_nominal, safe_set)

        # Find largest roa contained in the grid
        grid = self.discretization
        values = self.values.detach().cpu().numpy()
        values = values.reshape(grid.num_points)
        if len(grid.num_points) == 2:
            v1 = np.min(values[0,:])
            v2 = np.min(values[-1,:])
            v3 = np.min(values[:,0])
            v4 = np.min(values[:,-1])
            # print(v1, v2, v3, v4, c_max_tmp)
            c_max_tmp =  max(min(v1, v2, v3, v4, c_max_tmp), 1e-4)
        elif len(grid.num_points) == 3:
            v1 = np.min(values[0,:,:])
            v2 = np.min(values[-1,:,:])
            v3 = np.min(values[:,0,:])
            v4 = np.min(values[:,-1,:])
            v5 = np.min(values[:,:,0])
            v6 = np.min(values[:,:,-1])
            c_max_tmp =  max(min(v1, v2, v3, v4, v5, v6, c_max_tmp), 1e-4)
        elif len(grid.num_points) == 4:
            v1 = np.min(values[0,:,:,:])
            v2 = np.min(values[-1,:,:,:])
            v3 = np.min(values[:,0,:,:])
            v4 = np.min(values[:,-1,:,:])
            v5 = np.min(values[:,:,0,:])
            v6 = np.min(values[:,:,-1,:])
            v7 = np.min(values[:,:,:,0])
            v8 = np.min(values[:,:,:,-1])
            # print(v1, v2, v3, v4, v5, v6, v7, v8, c_max_tmp)
            c_max_tmp =  max(min(v1, v2, v3, v4, v5, v6, v7, v8, c_max_tmp), 1e-4)
        else:
            raise ValueError("No matching state dim!")
        largest_safe_set = self.values <= c_max_tmp
        setattr(self, 'largest_safe_set_' + true_or_nominal, largest_safe_set.detach().cpu().numpy().ravel())
        setattr(self, 'c_max_' + true_or_nominal, c_max_tmp)
        
        
    def update_exp_stable_set(self, alpha, true_or_nominal, roa_true):
        if true_or_nominal != "true" and true_or_nominal != "nominal":
            raise ValueError("Have to choose between true or nominal!")
        closed_loop_dynamics = getattr(self, 'closed_loop_dynamics_' + true_or_nominal)

        exp_stable_set =  np.zeros(np.prod(self.discretization.num_points), dtype=bool)
        if self.initial_safe_set is not None:
                exp_stable_set[self.initial_safe_set] = True
        # # Assume cannot shrink
        # exp_stable_set = np.logical_or(exp_stable_set, getattr(self, 'exp_stable_set_' + true_or_nominal))
        
        self.update_values()
        value_order = np.argsort(np.squeeze(self.values.detach().cpu().numpy())) # ordered indices based on the values of the Lyapunov function
        exp_stable_set = exp_stable_set[value_order]
        roa_true_ordered = roa_true[value_order]

        # Verify safety in batches
        batch_size = config.batch_size
        batch_generator = batchify((value_order, exp_stable_set, roa_true_ordered),
                                   batch_size)
        index_to_state = self.discretization.index_to_state

        #######################################################################

        for i, (indices, safe_batch, roa_true_batch) in batch_generator:
            states = index_to_state(indices)
            # Update the safety with the safe_batch result
            
            thresh = torch.tensor(self.decrease_thresh, dtype=config.ptdtype, device=config.device) \
                if self.decrease_thresh is not None \
                else self.threshold(torch.tensor(states, dtype=config.ptdtype), self.tau)
            dot_vnn = lambda x: torch.sum(torch.mul(self.grad_lyapunov_function(x), closed_loop_dynamics(x)), dim=1)
            decrease = torch.add(dot_vnn(states), \
                alpha*torch.pow(torch.norm(torch.tensor(states, dtype=config.ptdtype,\
                device=config.device), p=2, dim = 1),2)).reshape(-1, 1)
            exp_stable = torch.squeeze(torch.lt(decrease, thresh)).detach().cpu().numpy()
            safe_batch |= exp_stable
            safe_batch &= roa_true_batch

            # Boolean array: argmin returns first element that is False
            # If all are safe then it returns 0
            bound = np.argmin(safe_batch)

            # Check if there are unsafe elements in the batch
            if bound > 0 or not safe_batch[0]:
                # Make sure all following points are labeled as unsafe (because the batch is ordered with the values of the Lyapunov function)
                safe_batch[bound:] = False
                break
        # The largest index of a safe value
        max_index = max(i + bound - 1, 0) # i is the starting index of each batch and bound is the index inside a batch
        # Set placeholder for c_max to the corresponding value
        c_max_exp_tmp = self.values[value_order[max_index]].detach().cpu().numpy().item()
        setattr(self, 'c_max_exp_unconstrained_' + true_or_nominal, c_max_exp_tmp)
        # Restore the order of the safe set
        safe_nodes = value_order[exp_stable_set]
        exp_stable_set[:] = False
        exp_stable_set[safe_nodes] = True

        # Ensure the initial safe set is kept
        if self.initial_safe_set is not None:
            exp_stable_set[self.initial_safe_set] = True

        setattr(self, 'exp_stable_set_' + true_or_nominal, exp_stable_set)
        # Find largest roa contained in the grid
        grid = self.discretization
        values = self.values.detach().cpu().numpy()
        values = values.reshape(grid.num_points)
        if len(grid.num_points) == 2:
            v1 = np.min(values[0,:])
            v2 = np.min(values[-1,:])
            v3 = np.min(values[:,0])
            v4 = np.min(values[:,-1])
            c_max_exp_tmp =  max(min(v1, v2, v3, v4, c_max_exp_tmp), 1e-4)
        elif len(grid.num_points) == 3:
            v1 = np.min(values[0,:,:])
            v2 = np.min(values[-1,:,:])
            v3 = np.min(values[:,0,:])
            v4 = np.min(values[:,-1,:])
            v5 = np.min(values[:,:,0])
            v6 = np.min(values[:,:,-1])
            c_max_exp_tmp =  max(min(v1, v2, v3, v4, v5, v6, c_max_exp_tmp), 1e-4)
        elif len(grid.num_points) == 4:
            v1 = np.min(values[0,:,:,:])
            v2 = np.min(values[-1,:,:,:])
            v3 = np.min(values[:,0,:,:])
            v4 = np.min(values[:,-1,:,:])
            v5 = np.min(values[:,:,0,:])
            v6 = np.min(values[:,:,-1,:])
            v7 = np.min(values[:,:,:,0])
            v8 = np.min(values[:,:,:,-1])
            c_max_exp_tmp =  max(min(v1, v2, v3, v4, v5, v6, v7, v8, c_max_exp_tmp), 1e-4)
        else:
            raise ValueError("No matching state dim!")
        largest_exp_stable_set = self.values <= c_max_exp_tmp
        setattr(self, 'largest_exp_stable_set_' + true_or_nominal, largest_exp_stable_set.detach().cpu().numpy().ravel())
        setattr(self, 'c_max_exp_' + true_or_nominal, c_max_exp_tmp)
        

