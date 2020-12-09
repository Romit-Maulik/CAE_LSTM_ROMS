import numpy as np

# Global parameters of WENO SWE
Nx = 64      # Discretization in x direction
Ny = 64      # Discretization in y direction
ft = 0.5     # Final time
Lx = 1.0     # Length of domain in x direction
Ly = 1.0     # Length of domain in y direction
rho = 1.0          # Density of fluid [kg/m^3)]
dt = 0.001  # discrete timestep
dx = Lx/Nx
dy = Ly/Ny
grav = 9.8
Nt = int(ft/dt)

# Global parameters of SWE ROM
K = 40

# Mode of solver
fvm_solve = False # True for data generation, False for POD-GP
plot_viz = False
num_steps_per_plot = 5 # The number of timesteps per snapshot
num_samples = 100 # Total number of simulations
num_train = 90 # number of training simulations

if __name__ == '__main__':
    print('This is the parameter file')
    print(dx)

    data = np.load('PCA_Coefficients_q1_pred.npy')
    print(data.shape)
    data = np.load('PCA_Coefficients_q1_test.npy')
    print(data.shape)
    data = np.load('PCA_Coefficients_q1_train.npy')
    print(data.shape)
    
