import numpy as np

class system_properties(object):
    def __init__(self, pendulum_dict):
        for k in pendulum_dict.keys():
            exec("self.{} = pendulum_dict[k]".format(k, k) )



# Evey dynamical system is defined as a dictionary
all_systems = {}
# Pendulum 1
state_dim = 2
action_dim = 1
theta_max = np.deg2rad(180 * 1.0)                     # angular position [rad]
omega_max = np.deg2rad(180 * 1.0)                     # angular velocity [rad/s]
state_norm = (theta_max, omega_max)
system_dict = { "type": "pendulum",
                "state_dim": 2,
                "action_dim": 1,
                "g":9.81, 
                "m":1,
                "L":0.5, 
                "b":0.0,
                "state_norm":state_norm,
                }
u_max = system_dict["g"] * system_dict["m"] * system_dict["L"]  # torque [N.m], control action
system_dict["action_norm"]  = (u_max,)
all_systems["pendulum1"] = system_properties(system_dict)

# Pendulum 1 nominal
state_dim = 2
action_dim = 1
system_dict = { "type": "pendulum",
                "state_dim": 2,
                "action_dim": 1,
                "g":9.81, 
                "m":1*0.8,
                "L":0.5*0.8, 
                "b":0.0,
                "state_norm":all_systems["pendulum1"].state_norm,
                }

system_dict["action_norm"]  = all_systems["pendulum1"].action_norm
all_systems["pendulum1_nominal"] = system_properties(system_dict)

# Pendulum 2
state_dim = 2
action_dim = 1
theta_max = np.deg2rad(180 * 1.0)                     # angular position [rad]
omega_max = np.deg2rad(180 * 1.0)                     # angular velocity [rad/s]
state_norm = (theta_max, omega_max)
system_dict = { "type": "pendulum",
                "state_dim": 2,
                "action_dim": 1,
                "g":9.81, 
                "m":1,
                "L":0.5, 
                "b":0.0,
                "state_norm":state_norm,
                }
u_max = 10 
system_dict["action_norm"]  = (u_max,)
all_systems["pendulum2"] = system_properties(system_dict)

# Pendulum 1 nominal
state_dim = 2
action_dim = 1
system_dict = { "type": "pendulum",
                "state_dim": 2,
                "action_dim": 1,
                "g":9.81, 
                "m":1*0.8,
                "L":0.5*0.8, 
                "b":0.0,
                "state_norm":all_systems["pendulum2"].state_norm,
                }

system_dict["action_norm"]  = all_systems["pendulum2"].action_norm
all_systems["pendulum2_nominal"] = system_properties(system_dict)

# # Pendulum 2
# state_dim = 2
# action_dim = 1
# theta_max = np.deg2rad(180)                     # angular position [rad]
# omega_max = np.deg2rad(360)                     # angular velocity [rad/s]
# state_norm = (theta_max, omega_max)
# system_dict = { "type": "pendulum",
#                 "state_dim": 2,
#                 "action_dim": 1,
#                 "g":9.81, 
#                 "m":0.25,
#                 "L":0.5, 
#                 "b":0.1,
#                 "state_norm":state_norm, 
#                 }
# u_max = system_dict["g"] * system_dict["m"] * system_dict["L"] * np.sin(np.deg2rad(60))  # torque [N.m], control action
# system_dict["action_norm"]  = (u_max,)
# all_systems["pendulum2"] = system_properties(system_dict)

# Vanderpol
state_dim = 2
action_dim = 1
x_max     = 10                   # linear position [m]
y_max     = 10                  # angular position [rad]
state_norm = (x_max, y_max)
action_norm = None
system_dict = { "type": "vanderpol",
                "state_dim": 2,
                "action_dim": None,
                "damping":3.0,
                "state_norm":state_norm, 
                "action_norm":action_norm,
                }
all_systems["vanderpol"] = system_properties(system_dict)


# Backstepping 3D 1
state_dim = 3
action_dim = 1
x1_max = np.float64(2)
x2_max = np.float64(2)
x3_max = np.float64(2)

state_norm = (x1_max, x2_max, x3_max)
system_dict = { "type": "backstepping_3d",
                "state_dim": 3,
                "action_dim": 1,
                "a":1, 
                "b":1,
                "c":1,
                "d":1,
                "state_norm":state_norm,
                }
u_max = np.float64(10)
system_dict["action_norm"]  = (u_max,)
all_systems["backstepping_3d_1"] = system_properties(system_dict)

# Backstepping 3D 1 nominal
state_dim = 3
action_dim = 1
x1_max = np.float64(2)
x2_max = np.float64(2)
x3_max = np.float64(2)

state_norm = (x1_max, x2_max, x3_max)
system_dict = { "type": "backstepping_3d",
                "state_dim": 3,
                "action_dim": 1,
                "a":0.9, 
                "b":0.8,
                "c":0.9,
                "d":0.8,
                "state_norm":state_norm,
                }
u_max = np.float64(10)
system_dict["action_norm"]  = (u_max,)
all_systems["backstepping_3d_1_nominal"] = system_properties(system_dict)

# Backstepping 3D 2
state_dim = 3
action_dim = 1
x1_max = np.float64(1.5)
x2_max = np.float64(1.5)
x3_max = np.float64(2)

state_norm = (x1_max, x2_max, x3_max)
system_dict = { "type": "backstepping_3d",
                "state_dim": 3,
                "action_dim": 1,
                "a":1, 
                "b":1,
                "c":1,
                "d":1,
                "state_norm":state_norm,
                }
u_max = np.float64(10)
system_dict["action_norm"]  = (u_max,)
all_systems["backstepping_3d_2"] = system_properties(system_dict)

# Backstepping 3D 2 nominal
state_dim = 3
action_dim = 1

system_dict = { "type": "backstepping_3d",
                "state_dim": 3,
                "action_dim": 1,
                "a":0.9, 
                "b":0.8,
                "c":0.9,
                "d":0.8,
                "state_norm":state_norm,
                }
u_max = np.float64(10)
system_dict["action_norm"]  = all_systems["backstepping_3d_2"].action_norm
system_dict["state_norm"]  = all_systems["backstepping_3d_2"].state_norm
all_systems["backstepping_3d_2_nominal"] = system_properties(system_dict)


# Cartpole 1
state_dim = 4
action_dim = 1
x_max = np.float64(1)
theta_max = np.float64(np.pi/4)
v_max = np.float64(2)
omega_max = np.float64(2)

state_norm = (x_max, theta_max, v_max, omega_max) # Match the order of the states !!!
system_dict = { "type": "cartpole",
                "state_dim": 4,
                "action_dim": 1,
                "M":1, 
                "m":0.3,
                "g":9.81,
                "l":1,
                "b":0,
                "state_norm":state_norm,
                }
u_max = np.float64(10)
system_dict["action_norm"]  = (u_max,)
all_systems["cartpole1"] = system_properties(system_dict)

# Cartpole 1 nominal
state_dim = 4
action_dim = 1
system_dict = { "type": "cartpole",
                "state_dim": 4,
                "action_dim": 1,
                "M":1*0.8, 
                "m":0.3*0.9,
                "g":9.81,
                "l":1*0.8,
                "b":0,
                }
u_max = np.float64(10)
system_dict["action_norm"]  = all_systems["cartpole1"].action_norm
system_dict["state_norm"]  = all_systems["cartpole1"].state_norm
all_systems["cartpole1_nominal"] = system_properties(system_dict)

# Cartpole 2
state_dim = 4
action_dim = 1
x_max = np.float64(1)
theta_max = np.float64(np.pi/6)
v_max = np.float64(1.5)
omega_max = np.float64(1)

state_norm = (x_max, theta_max, v_max, omega_max) # Match the order of the states !!!
system_dict = { "type": "cartpole",
                "state_dim": 4,
                "action_dim": 1,
                "M":1, 
                "m":0.3,
                "g":9.81,
                "l":1,
                "b":0,
                "state_norm":state_norm,
                }
u_max = np.float64(10)
system_dict["action_norm"]  = (u_max,)
all_systems["cartpole2"] = system_properties(system_dict)

# Cartpole 2 nominal
state_dim = 4
action_dim = 1
system_dict = { "type": "cartpole",
                "state_dim": 4,
                "action_dim": 1,
                "M":1*0.8, 
                "m":0.3*0.9,
                "g":9.81,
                "l":1*0.8,
                "b":0,
                }
u_max = np.float64(10)
system_dict["action_norm"]  = all_systems["cartpole2"].action_norm
system_dict["state_norm"]  = all_systems["cartpole2"].state_norm
all_systems["cartpole2_nominal"] = system_properties(system_dict)

# Cartpole 2 perturbed
state_dim = 4
action_dim = 1
system_dict = { "type": "cartpole",
                "state_dim": 4,
                "action_dim": 1,
                "M":1, 
                "m":0.3,
                "g":9.81,
                "l":1,
                "b":9.05, # 9.1 for plots in exp_41_keep_eg3
                }
u_max = np.float64(10)
system_dict["action_norm"]  = all_systems["cartpole2"].action_norm
system_dict["state_norm"]  = all_systems["cartpole2"].state_norm
all_systems["cartpole2_perturbed"] = system_properties(system_dict)

# Euler equation for a rotating rigid spacecraft
state_dim = 3
action_dim = 3
x1_max = np.float64(np.pi)
x2_max = np.float64(np.pi)
x3_max = np.float64(np.pi)

state_norm = (x1_max, x2_max, x3_max) # Match the order of the states !!!
system_dict = { "type": "euler_equation_3d",
                "state_dim": state_dim,
                "action_dim": action_dim,
                "J1":1, 
                "J2":1,
                "J3":1,
                "state_norm":state_norm,
                }
u1_max = np.float64(10)
u2_max = np.float64(10)
u3_max = np.float64(10)
system_dict["action_norm"]  = (u1_max, u2_max, u3_max)
all_systems["euler_equation1"] = system_properties(system_dict)

# Euler equation for a rotating rigid spacecraft nominal
state_dim = 3
action_dim = 3
system_dict = { "type": "euler_equation_3d",
                "state_dim": 4,
                "action_dim": 1,
                "J1":1 * 0.75, 
                "J2":1 * 0.8,
                "J3":1 * 0.85,
                }
system_dict["action_norm"]  = all_systems["euler_equation1"].action_norm
system_dict["state_norm"]  = all_systems["euler_equation1"].state_norm
all_systems["euler_equation1_nominal"] = system_properties(system_dict)
