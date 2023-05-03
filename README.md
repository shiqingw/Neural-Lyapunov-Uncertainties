# Neural Lyapunov Control for Nonlinear Systems with Unstructured Uncertainties

<em>This GitHub repository contains the codebase for the paper Wei, S., Krishnamurthy, P., & Khorrami, F. (2023). Neural Lyapunov Control for Nonlinear Systems with Unstructured Uncertainties. arXiv preprint arXiv:2303.09678, which is also accepted at the 2023 American Control Conference (ACC).</em>

**Note: Part of the code is adapted from https://github.com/amehrjou/neural_lyapunov_redesign**

Table of Contents:
1. [Prerequisites](#prerequisites)
2. [Code Explained](#code-explained)
3. [Code Usage](#code-usage)

## Prerequisites
The code is tested on
- Python 3.8.15
- PyTorch 1.13.0
- dReal4

## Code Explained
The codebase contains three examples: the inverted pendulum (`eg1_inverted_pendulum_2d`), a hypothetical system of strict feedback form (`eg2_backstepping_3d`), and the cart-pole (`eg3_cartpole_4d`).

## Code Usage
1. Open the main script, e.g. `eg1_inverted_pendulum_2d/inv_pend_2d_sum4_nn_controller.py` and modify the hyper-parameters
2. Execute the script
3. After training, the results will be saved in the `results/exp_XXX` where XXX is the `exp_num` you defined in the hyper-parameters

