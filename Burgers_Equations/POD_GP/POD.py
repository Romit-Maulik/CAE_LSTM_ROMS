import numpy as np
from numpy import linalg as LA

def generate_pod_bases(snapshot_matrix,num_modes,tsteps): #Mean removed
    '''
    Takes input of a snapshot matrix and computes POD bases
    Outputs truncated POD bases and coefficients
    '''
    new_mat = np.matmul(np.transpose(snapshot_matrix),snapshot_matrix)

    w,v = LA.eig(new_mat)

    # Bases
    phi = np.real(np.matmul(snapshot_matrix,v))
    
    trange = np.arange(np.shape(tsteps)[0])
    phi[:,trange] = phi[:,trange]/np.sqrt(w[:])

    coefficient_matrix = np.matmul(np.transpose(phi),snapshot_matrix)

    # Truncate coefficient and phi matrices
    phi_trunc = phi[:,0:num_modes] # Columns are modes
    cf_trunc = coefficient_matrix[0:num_modes,:] #Columns are time, rows are modal coefficients

    return phi_trunc, cf_trunc

def plot_pod_modes(phi,mode_num):
    plt.figure()
    plt.plot(phi[:,mode_num])
    plt.show()
