# This initializes the problem class for SWE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from parameters import Nx, Ny, Lx, Ly
from parameters import rho, grav, dt, dx, dy, ft
from parameters import K
from parameters import plot_viz, num_steps_per_plot, num_samples, num_train

# Common functions for spatial discretizations
def state_reconstruction(q,Nx,Ny):
    # Weno5
    pad = 3
    qtemp = periodic_bc(q,pad)

    # Smoothness indicators in x
    beta_0 = 13.0/12.0*(qtemp[pad-2:pad+Nx-2,:]-2.0*qtemp[pad-1:pad+Nx-1,:]+qtemp[pad:Nx+pad,:])**2 \
            + 1.0/4.0*(qtemp[pad-2:pad+Nx-2,:]-4.0*qtemp[pad-1:pad+Nx-1,:]+3.0*qtemp[pad:Nx+pad,:])**2
    
    beta_1 = 13.0/12.0*(qtemp[pad-1:pad+Nx-1,:]-2.0*qtemp[pad:pad+Nx,:]+qtemp[pad+1:Nx+pad+1,:])**2 \
            + 1.0/4.0*(qtemp[pad-1:pad+Nx-1,:]-qtemp[pad+1:pad+Nx+1,:])**2
    
    beta_2 = 13.0/12.0*(qtemp[pad:pad+Nx,:]-2.0*qtemp[pad+1:pad+Nx+1,:]+qtemp[pad+2:Nx+pad+2,:])**2 \
            + 1.0/4.0*(3.0*qtemp[pad:pad+Nx,:]-4.0*qtemp[pad+1:pad+Nx+1,:]+qtemp[pad+2:Nx+pad+2,:])**2

    # nonlinear weights in x
    alpha_0 = (1.0/10.0)/((beta_0+1.0e-6)**2)
    alpha_1 = (6.0/10.0)/((beta_1+1.0e-6)**2)
    alpha_2 = (3.0/10.0)/((beta_2+1.0e-6)**2)

    # Find nonlinear weights
    w_0 = (alpha_0/(alpha_0+alpha_1+alpha_2))/6.0
    w_1 = (alpha_1/(alpha_0+alpha_1+alpha_2))/6.0
    w_2 = (alpha_2/(alpha_0+alpha_1+alpha_2))/6.0

    # Find state reconstructions in x - wave to right (at i+1/2)
    qxright = w_0*(2.0*qtemp[pad-2:pad+Nx-2,:]-7.0*qtemp[pad-1:pad+Nx-1,:]+11.0*qtemp[pad:pad+Nx,:]) \
          + w_1*(-qtemp[pad-1:pad+Nx-1,:]+5.0*qtemp[pad:pad+Nx,:]+2.0*qtemp[pad+1:pad+Nx+1,:]) \
          + w_2*(2.0*qtemp[pad:pad+Nx,:]+5.0*qtemp[pad+1:pad+Nx+1,:]-qtemp[pad+2:pad+Nx+2,:])

    # Find state reconstructions in x - wave to left (at i+1/2)
    qxleft = w_0*(2.0*qtemp[pad+2:pad+Nx+2,:]-7.0*qtemp[pad+1:pad+Nx+1,:]+11.0*qtemp[pad:pad+Nx,:]) \
          + w_1*(-qtemp[pad+1:pad+Nx+1,:]+5.0*qtemp[pad:pad+Nx,:]+2.0*qtemp[pad-1:pad+Nx-1,:]) \
          + w_2*(2.0*qtemp[pad:pad+Nx,:]+5.0*qtemp[pad-1:pad+Nx-1,:]-qtemp[pad-2:pad+Nx-2,:])

    qxleft = qxleft[:,pad:pad+Ny]
    qxright = qxright[:,pad:pad+Ny]

    # Smoothness indicators in y
    beta_0 = 13.0/12.0*(qtemp[:,pad-2:pad+Ny-2]-2.0*qtemp[:,pad-1:pad+Ny-1]+qtemp[:,pad:Ny+pad])**2 \
            + 1.0/4.0*(qtemp[:,pad-2:pad+Ny-2]-4.0*qtemp[:,pad-1:pad+Ny-1]+3.0*qtemp[:,pad:Ny+pad])**2
    
    beta_1 = 13.0/12.0*(qtemp[:,pad-1:pad+Ny-1]-2.0*qtemp[:,pad:pad+Ny]+qtemp[:,pad+1:Ny+pad+1])**2 \
            + 1.0/4.0*(qtemp[:,pad-1:pad+Ny-1]-qtemp[:,pad+1:pad+Ny+1])**2
    
    beta_2 = 13.0/12.0*(qtemp[:,pad:pad+Ny]-2.0*qtemp[:,pad+1:pad+Ny+1]+qtemp[:,pad+2:Ny+pad+2])**2 \
            + 1.0/4.0*(3.0*qtemp[:,pad:pad+Ny]-4.0*qtemp[:,pad+1:pad+Ny+1]+qtemp[:,pad+2:Ny+pad+2])**2

    # nonlinear weights in x
    alpha_0 = (1.0/10.0)/((beta_0+1.0e-6)**2)
    alpha_1 = (6.0/10.0)/((beta_1+1.0e-6)**2)
    alpha_2 = (3.0/10.0)/((beta_2+1.0e-6)**2)

    # Find nonlinear weights
    w_0 = (alpha_0/(alpha_0+alpha_1+alpha_2))/6.0
    w_1 = (alpha_1/(alpha_0+alpha_1+alpha_2))/6.0
    w_2 = (alpha_2/(alpha_0+alpha_1+alpha_2))/6.0

    # Find state reconstructions in y - qright (at i+1/2)
    qyright = w_0*(2.0*qtemp[:,pad-2:pad+Ny-2]-7.0*qtemp[:,pad-1:pad+Ny-1]+11.0*qtemp[:,pad:pad+Ny]) \
          + w_1*(-qtemp[:,pad-1:pad+Ny-1]+5.0*qtemp[:,pad:pad+Ny]+2.0*qtemp[:,pad+1:pad+Ny+1]) \
          + w_2*(2.0*qtemp[:,pad:pad+Ny]+5.0*qtemp[:,pad+1:pad+Ny+1]-qtemp[:,pad+2:pad+Ny+2])

    # Find state reconstructions in y - wave to left (at i+1/2)
    qyleft = w_0*(2.0*qtemp[:,pad+2:pad+Ny+2]-7.0*qtemp[:,pad+1:pad+Ny+1]+11.0*qtemp[:,pad:pad+Ny]) \
          + w_1*(-qtemp[:,pad+1:pad+Ny+1]+5.0*qtemp[:,pad:pad+Ny]+2.0*qtemp[:,pad-1:pad+Ny-1]) \
          + w_2*(2.0*qtemp[:,pad:pad+Ny]+5.0*qtemp[:,pad-1:pad+Ny-1]-qtemp[:,pad-2:pad+Ny-2])

    qyleft = qyleft[pad:pad+Nx,:]
    qyright = qyright[pad:pad+Nx,:]

    return qxleft, qxright, qyleft, qyright

def reimann_solve(spec_rad,fl,fr,ql,qr,dim):
    # Rusanov reimann solver
    pad = 3
    srt = periodic_bc(spec_rad,pad)
    if dim == 'x':
        srt = np.maximum.reduce([srt[pad-3:Nx+pad-3,pad:Ny+pad],srt[pad-2:Nx+pad-2,pad:Ny+pad],srt[pad-1:Nx+pad-1,pad:Ny+pad],\
            srt[pad:Nx+pad,pad:Ny+pad],srt[pad+1:Nx+pad+1,pad:Ny+pad],srt[pad+2:Nx+pad+2,pad:Ny+pad],srt[pad+3:Nx+pad+3,pad:Ny+pad]])
        flux = 0.5*(fr+fl) + 0.5*srt*(qr+ql)
        return flux
    else:
        srt = np.maximum.reduce([srt[pad:Nx+pad,pad-3:Ny+pad-3],srt[pad:Nx+pad,pad-2:Ny+pad-2],srt[pad:Nx+pad,pad-1:Ny+pad-1],\
            srt[pad:Nx+pad,pad:Ny+pad],srt[pad:Nx+pad,pad+1:Ny+pad+1],srt[pad:Nx+pad,pad+2:Ny+pad+2],srt[pad:Nx+pad,pad+3:Ny+pad+3]])
        flux = 0.5*(fr+fl) + 0.5*srt*(qr+ql)
        return flux

def periodic_bc(q,pad):
    qtemp = np.zeros(shape=(q.shape[0]+2*pad,q.shape[1]+2*pad),dtype='double')
    # Periodicity updates
    qtemp[pad:Nx+pad,pad:Ny+pad] = q[:,:]
    # x direction periodicity
    qtemp[0:pad,:] = qtemp[Nx-pad:Nx,:]
    qtemp[Nx+pad:,:] = qtemp[pad:2*pad,:]
    # y direction periodicity
    qtemp[:,0:pad] = qtemp[:,Ny-pad:Ny]
    qtemp[:,Ny+pad:] = qtemp[:,pad:2*pad]

    return qtemp

def spectral_radius(q1,q2):
    sound_speed = 2.0*np.sqrt(q1/rho*grav)
    u = q2/q1
    return np.maximum.reduce([np.abs(u+sound_speed),np.abs(u-sound_speed),\
                       np.abs(sound_speed)])

def flux_reconstruction(q1,q2,q3):
    spec_rad_x = spectral_radius(q1,q2)
    spec_rad_y = spectral_radius(q1,q3)

    q1xleft, q1xright, q1yleft, q1yright = state_reconstruction(q1,Nx,Ny)
    q2xleft, q2xright, q2yleft, q2yright = state_reconstruction(q2,Nx,Ny)
    q3xleft, q3xright, q3yleft, q3yright = state_reconstruction(q3,Nx,Ny)

    # Reconstructing fluxes for q1
    f1xleft = np.copy(q2xleft)
    f1xright = np.copy(q2xright)
    f1x = reimann_solve(spec_rad_x,f1xleft,f1xright,q1xleft,q1xright,'x')

    f1yleft = np.copy(q3yleft)
    f1yright = np.copy(q3yright)
    f1y = reimann_solve(spec_rad_y,f1yleft,f1yright,q1yleft,q1yright,'y')

    # Reconstructing fluxes for q2
    f2xleft = (q2xleft**2)/(q1xleft) + 0.5*(q1xleft**2)*(grav/rho)
    f2xright = (q2xright**2)/(q1xright) + 0.5*(q1xright**2)*(grav/rho)
    f2x = reimann_solve(spec_rad_x,f1xleft,f2xright,q2xleft,q2xright,'x')

    f2yleft = (q2yleft*q3yleft/q1yleft)
    f2yright = (q2yright*q3yright/q1yright)
    f2y = reimann_solve(spec_rad_y,f2yleft,f2yright,q2yleft,q2yright,'y')

    # Reconstructing fluxes for q3
    f3xleft = (q2xleft*q3xleft/q1xleft)
    f3xright = (q2xright*q3xright/q1xright)
    f3x = reimann_solve(spec_rad_x,f3xleft,f3xright,q3xleft,q3xright,'x')

    f3yleft = (q3yleft**2)/(q1yleft) + 0.5*(q1yleft**2)*(grav/rho)
    f3yright = (q3yright**2)/(q1yright) + 0.5*(q1yright**2)*(grav/rho)
    f3y = reimann_solve(spec_rad_y,f3yleft,f3yright,q3yleft,q3yright,'y')

    return f1x, f1y, f2x, f2y, f3x, f3y

# Plotting functions
def plot_coefficients(Ytilde):
    fig,ax = plt.subplots(nrows=1,ncols=4)
    ax[0].plot(Ytilde[0,:],label='Mode 1')
    ax[1].plot(Ytilde[1,:],label='Mode 2')
    ax[2].plot(Ytilde[2,:],label='Mode 3')
    ax[3].plot(Ytilde[3,:],label='Mode 4')
    plt.legend()
    plt.show()

def plot_fields_debug(X,Y,q,label,iter):
    fig = plt.figure(figsize = (11, 7))
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, q, rstride = 1, cstride = 1,
        cmap = plt.cm.jet, linewidth = 0, antialiased = True)

    ax.set_title('Visualization', fontname = "serif", fontsize = 17)
    ax.set_xlabel("x [m]", fontname = "serif", fontsize = 16)
    ax.set_ylabel("y [m]", fontname = "serif", fontsize = 16)

    if label == 'q1':
        ax.set_zlim((0,2))
    elif label == 'q2':
        ax.set_zlim((-1,1))
    else:
        ax.set_zlim((-1,1))
    plt.savefig(label+'_'+str(iter)+'.png')

# Shallow water equations class
class shallow_water(object):
    """docstring for ClassName"""
    def __init__(self,args=[0,0]):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly

        x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)  # Array with x-points
        y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)  # Array with y-points

        # Meshgrid for plotting
        self.X, self.Y = np.meshgrid(x, y) 

        # Initialize fields
        self.initialize(args) 

        # Field storage for viz
        self.q_list = []

        # Plot interval
        self.plot_interval = num_steps_per_plot

        # Field storage for ROM
        self.snapshots_pod = []   # at plot interval

    def initialize(self,args=[0,0]):
        loc_x = args[0]
        loc_y = args[1]

        # There are three conserved quantities - initialize
        self.q1 = 1.0+(rho*np.exp(-((self.X-loc_x)**2/(2*(0.05)**2) + (self.Y-loc_y)**2/(2*(0.05)**2))))
        self.q2 = np.zeros(shape=(self.Nx,self.Ny),dtype='double')
        self.q3 = np.zeros(shape=(self.Nx,self.Ny),dtype='double')
        
    def right_hand_side(self,q1,q2,q3):
        f1x, f1y, f2x, f2y, f3x, f3y = flux_reconstruction(q1,q2,q3) # these are all i+1/2

        # Periodicity
        pad = 1
        f1xtemp = periodic_bc(f1x,pad)
        f1ytemp = periodic_bc(f1y,pad)
        f2xtemp = periodic_bc(f2x,pad)
        f2ytemp = periodic_bc(f2y,pad)
        f3xtemp = periodic_bc(f3x,pad)
        f3ytemp = periodic_bc(f3y,pad)

        r1 = 1.0/dx*(f1xtemp[pad:Nx+pad,pad:Ny+pad]-f1xtemp[pad-1:Nx+pad-1,pad:Ny+pad]) + 1.0/dy*(f1ytemp[pad:Nx+pad,pad:Ny+pad]-f1ytemp[pad:Nx+pad,pad-1:Ny+pad-1])
        r2 = 1.0/dx*(f2xtemp[pad:Nx+pad,pad:Ny+pad]-f2xtemp[pad-1:Nx+pad-1,pad:Ny+pad]) + 1.0/dy*(f2ytemp[pad:Nx+pad,pad:Ny+pad]-f2ytemp[pad:Nx+pad,pad-1:Ny+pad-1])
        r3 = 1.0/dx*(f3xtemp[pad:Nx+pad,pad:Ny+pad]-f3xtemp[pad-1:Nx+pad-1,pad:Ny+pad]) + 1.0/dy*(f3ytemp[pad:Nx+pad,pad:Ny+pad]-f3ytemp[pad:Nx+pad,pad-1:Ny+pad-1])

        return -r1, -r2, -r3

    def integrate_rk(self):
        # Equally spaced time integration
        q1temp = np.copy(self.q1)
        q2temp = np.copy(self.q2)
        q3temp = np.copy(self.q3)

        r1_k1, r2_k1, r3_k1 = self.right_hand_side(q1temp,q2temp,q3temp) # Note switch in sign
               
        q1temp[:,:] = self.q1[:,:] + dt*(r1_k1[:,:])
        q2temp[:,:] = self.q2[:,:] + dt*(r2_k1[:,:])
        q3temp[:,:] = self.q3[:,:] + dt*(r3_k1[:,:])
       
        r1_k2, r2_k2, r3_k2 = self.right_hand_side(q1temp,q2temp,q3temp) # Note switch in sign

        q1temp[:,:] = self.q1[:,:] + 0.125*dt*r1_k1[:,:] + 0.125*dt*r1_k2[:,:]
        q2temp[:,:] = self.q2[:,:] + 0.125*dt*r2_k1[:,:] + 0.125*dt*r2_k2[:,:]
        q3temp[:,:] = self.q3[:,:] + 0.125*dt*r3_k1[:,:] + 0.125*dt*r3_k2[:,:]
       
        r1_k3, r2_k3, r3_k3 = self.right_hand_side(q1temp,q2temp,q3temp) # Note switch in sign
        
        self.q1[:,:] = self.q1[:,:] + (1.0/6.0)*dt*r1_k1[:,:] + (1.0/6.0)*dt*r1_k2[:,:] + (2.0/3.0)*dt*r1_k3[:,:]
        self.q2[:,:] = self.q2[:,:] + (1.0/6.0)*dt*r2_k1[:,:] + (1.0/6.0)*dt*r2_k2[:,:] + (2.0/3.0)*dt*r2_k3[:,:]
        self.q3[:,:] = self.q3[:,:] + (1.0/6.0)*dt*r3_k1[:,:] + (1.0/6.0)*dt*r3_k2[:,:] + (2.0/3.0)*dt*r3_k3[:,:]

    def solve(self):
        self.t = 0
        plot_iter = 0
        save_iter = 0

        # Save initial conditions
        flattened_data = np.concatenate((self.q1.flatten(),self.q2.flatten(),self.q3.flatten()),axis=0)
        self.snapshots_pod.append(flattened_data)

        while self.t < ft:            
            print('Time is:',self.t)
            self.t = self.t + dt
            self.integrate_rk()
            
            if plot_iter == self.plot_interval:
                # Save snapshots
                flattened_data = np.concatenate((self.q1.flatten(),self.q2.flatten(),self.q3.flatten()),axis=0)
                self.snapshots_pod.append(flattened_data)
                
                if plot_viz:
                    plot_fields_debug(self.X,self.Y,self.q1,'q1',save_iter)
                
                plot_iter = 0
                save_iter = save_iter + 1

            plot_iter = plot_iter + 1
            
        print('Solution finished')

class shallow_water_rom(object):
    def __init__(self,snapshot_matrix_pod,snapshot_matrix_test):
        """
        K - number of POD DOF for GP        
        snapshot_matrix_pod - At snapshot location
        """
        self.K = K

        self.q1_snapshot_matrix_pod = snapshot_matrix_pod[:Nx*Ny,:]
        self.q2_snapshot_matrix_pod = snapshot_matrix_pod[Nx*Ny:2*Nx*Ny,:]
        self.q3_snapshot_matrix_pod = snapshot_matrix_pod[2*Nx*Ny:,:]

        self.q1_snapshot_matrix_test = snapshot_matrix_test[:Nx*Ny,:]
        self.q2_snapshot_matrix_test = snapshot_matrix_test[Nx*Ny:2*Nx*Ny,:]
        self.q3_snapshot_matrix_test = snapshot_matrix_test[2*Nx*Ny:,:]

        # Plot interval
        self.plot_interval = num_steps_per_plot

        # Plot related
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly

        x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)  # Array with x-points
        y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)  # Array with y-points

        # Meshgrid for plotting
        self.X, self.Y = np.meshgrid(x, y) 

    def method_of_snapshots(self,snapshot_matrix_pod,snapshot_matrix_test):
        """
        Read snapshot_matrix (field or nonlinear term) and compute the POD bases and coefficients
        snapshot_matrix_pod - N x S - where N is DOF, S snapshots
        V - truncated POD basis matrix - shape: NxK - K is truncation number
        Ytilde - shape: KxS - POD basis coefficients for train data
        Ytilde_test - shape: KxS - POD basis coefficients for test data
        """
        new_mat = np.matmul(np.transpose(snapshot_matrix_pod),snapshot_matrix_pod)
        w,v = np.linalg.eig(new_mat)
        # Bases
        V = np.real(np.matmul(snapshot_matrix_pod,v)) 
        trange = np.arange(np.shape(V)[1])
        V[:,trange] = V[:,trange]/np.sqrt(w[:])
        # Truncate phis
        V = V[:,0:self.K] # Columns are modes
        # Find POD coefficients
        Ytilde = np.matmul(np.transpose(V),snapshot_matrix_pod)
        Ytilde_test = np.matmul(np.transpose(V),snapshot_matrix_test)

        return w, V, Ytilde, Ytilde_test

    def svd_method(self,snapshot_matrix_pod):
        """
        Read snapshot_matrix (field or nonlinear term) and compute the POD bases and coefficients
        snapshot_matrix_pod - N x S - where N is DOF, S snapshots
        V - truncated POD basis matrix - shape: NxK - K is truncation number
        Ytilde - shape: KxS - POD basis coefficients
        """
        phi, S, Vt = np.linalg.svd(snapshot_matrix_pod)
        Ytilde = np.matmul(phi.T[:,truncation],snapshot_matrix)
        Ytilde_test = np.matmul(phi.T[:,truncation],snapshot_matrix_test)

        return S, phi.T[:,self.K], Ytilde, Ytilde_test

    def generate_pod(self):
        # Do the POD of the conserved variables 
        self.q1_w, self.q1_V, self.q1_Ytilde, self.q1_Ytilde_test = self.method_of_snapshots(self.q1_snapshot_matrix_pod,self.q1_snapshot_matrix_test)
        self.q2_w, self.q2_V, self.q2_Ytilde, self.q2_Ytilde_test = self.method_of_snapshots(self.q2_snapshot_matrix_pod,self.q2_snapshot_matrix_test)
        self.q3_w, self.q3_V, self.q3_Ytilde, self.q3_Ytilde_test = self.method_of_snapshots(self.q3_snapshot_matrix_pod,self.q3_snapshot_matrix_test)

        # Print captured energy - using definition in https://arxiv.org/pdf/1308.3276.pdf
        print('Capturing ',np.sum(self.q1_w[0:self.K])/np.sum(self.q1_w),'% variance in conserved variable 1')
        print('Capturing ',np.sum(self.q2_w[0:self.K])/np.sum(self.q2_w),'% variance in conserved variable 2')
        print('Capturing ',np.sum(self.q3_w[0:self.K])/np.sum(self.q3_w),'% variance in conserved variable 3')

        np.save('PCA_Vectors_q1.npy',self.q1_V)  # The POD bases
        np.save('PCA_Vectors_q2.npy',self.q2_V) 
        np.save('PCA_Vectors_q3.npy',self.q3_V) 

        np.save('PCA_Coefficients_q1_train.npy',self.q1_Ytilde) # The true projection
        np.save('PCA_Coefficients_q2_train.npy',self.q2_Ytilde) 
        np.save('PCA_Coefficients_q3_train.npy',self.q3_Ytilde) 

        np.save('PCA_Coefficients_q1_test.npy',self.q1_Ytilde_test) # The true projection
        np.save('PCA_Coefficients_q2_test.npy',self.q2_Ytilde_test) 
        np.save('PCA_Coefficients_q3_test.npy',self.q3_Ytilde_test)

    def load_pregenerated_pod(self):

        self.q1_V = np.load('PCA_Vectors_q1.npy')  # The POD bases
        self.q2_V = np.load('PCA_Vectors_q2.npy') 
        self.q3_V = np.load('PCA_Vectors_q3.npy')

        self.q1_Ytilde = np.load('PCA_Coefficients_q1_train.npy') # The true projection
        self.q2_Ytilde = np.load('PCA_Coefficients_q2_train.npy') 
        self.q3_Ytilde = np.load('PCA_Coefficients_q3_train.npy') 

        self.q1_Ytilde_test = np.load('PCA_Coefficients_q1_test.npy') # The true projection
        self.q2_Ytilde_test = np.load('PCA_Coefficients_q2_test.npy')
        self.q3_Ytilde_test = np.load('PCA_Coefficients_q3_test.npy')

        if self.q1_V.shape[1] < self.K:
            print('Regenerate POD - not enough modes saved')
            exit()
        else:
            self.q1_V = self.q1_V[:,:self.K]
            self.q2_V = self.q2_V[:,:self.K]
            self.q3_V = self.q3_V[:,:self.K]

            self.q1_Ytilde = self.q1_Ytilde[:self.K]
            self.q2_Ytilde = self.q2_Ytilde[:self.K]
            self.q3_Ytilde = self.q3_Ytilde[:self.K]

            self.q1_Ytilde_test = self.q1_Ytilde_test[:self.K]
            self.q2_Ytilde_test = self.q2_Ytilde_test[:self.K]
            self.q3_Ytilde_test = self.q3_Ytilde_test[:self.K]

    def plot_reconstruction_error(self):
        fig,ax = plt.subplots(ncols=3)
        ax[0].plot(self.q1_w[:]/np.sum(self.q1_w))
        ax[1].plot(self.q2_w[:]/np.sum(self.q2_w))
        ax[2].plot(self.q3_w[:]/np.sum(self.q3_w))
        plt.show()

    def solve(self):
        from time import time
        num_test = int(num_samples-num_train)

        self.q1_snapshots = np.copy(self.q1_Ytilde_test)
        self.q2_snapshots = np.copy(self.q2_Ytilde_test)
        self.q3_snapshots = np.copy(self.q3_Ytilde_test)

        start_time = time()
        for test in range(num_test):

            plot_iter = 0
            save_iter = 0
            iter_num = 0

            # initalize solutions
            self.q1 = np.copy(self.q1_Ytilde_test[:,num_steps_per_plot*test])
            self.q2 = np.copy(self.q2_Ytilde_test[:,num_steps_per_plot*test])
            self.q3 = np.copy(self.q3_Ytilde_test[:,num_steps_per_plot*test])
            self.t = 0.0

            save_iter = num_steps_per_plot*test+1
            while self.t < ft:            
                print('Time is:',self.t)
                self.t = self.t + dt

                self.integrate_rk()
                iter_num = iter_num + 1
                
                if plot_iter == self.plot_interval:     
                    # q1_full = np.matmul(self.q1_V,self.q1)
                    # q2_full = np.matmul(self.q2_V,self.q2)
                    # q3_full = np.matmul(self.q3_V,self.q3)
                
                    # flattened_data = np.concatenate((q1_full,q2_full,q3_full),axis=0)
                    # self.rom_pred_snapshots.append(flattened_data)

                    self.q1_snapshots[:,save_iter] = self.q1[:]
                    self.q2_snapshots[:,save_iter] = self.q2[:]
                    self.q3_snapshots[:,save_iter] = self.q3[:]

                    
                    if plot_viz:
                        q1_full = np.reshape(q1_full,newshape=(Nx,Ny))
                        plot_fields_debug(self.X,self.Y,q1_full,'q1',save_iter)
                    
                    plot_iter = 0
                    save_iter = save_iter + 1
                
                plot_iter = plot_iter + 1

        print('Average elapsed time GP:',(time()-start_time)/(num_test))
        np.save('PCA_Coefficients_q1_pred.npy',self.q1_snapshots)
        np.save('PCA_Coefficients_q2_pred.npy',self.q2_snapshots)
        np.save('PCA_Coefficients_q3_pred.npy',self.q3_snapshots)
        

    def integrate_rk(self):
        # Equally spaced time integration in reduced space
        q1temp = np.copy(self.q1)
        q2temp = np.copy(self.q2)
        q3temp = np.copy(self.q3)

        r1_k1, r2_k1, r3_k1 = self.right_hand_side(q1temp,q2temp,q3temp) # Note switch in sign

        q1temp[:] = self.q1[:] + dt*(r1_k1[:])
        q2temp[:] = self.q2[:] + dt*(r2_k1[:])
        q3temp[:] = self.q3[:] + dt*(r3_k1[:])
        
        r1_k2, r2_k2, r3_k2 = self.right_hand_side(q1temp,q2temp,q3temp) # Note switch in sign
        
        q1temp[:] = self.q1[:] + 0.125*dt*r1_k1[:] + 0.125*dt*r1_k2[:]
        q2temp[:] = self.q2[:] + 0.125*dt*r2_k1[:] + 0.125*dt*r2_k2[:]
        q3temp[:] = self.q3[:] + 0.125*dt*r3_k1[:] + 0.125*dt*r3_k2[:]

       
        r1_k3, r2_k3, r3_k3 = self.right_hand_side(q1temp,q2temp,q3temp) # Note switch in sign
        
        self.q1[:] = self.q1[:] + (1.0/6.0)*dt*r1_k1[:] + (1.0/6.0)*dt*r1_k2[:] + (2.0/3.0)*dt*r1_k3[:]
        self.q2[:] = self.q2[:] + (1.0/6.0)*dt*r2_k1[:] + (1.0/6.0)*dt*r2_k2[:] + (2.0/3.0)*dt*r2_k3[:]
        self.q3[:] = self.q3[:] + (1.0/6.0)*dt*r3_k1[:] + (1.0/6.0)*dt*r3_k2[:] + (2.0/3.0)*dt*r3_k3[:]

    def right_hand_side(self,q1_red,q2_red,q3_red):
        """
        Function calculates nonlinear term using state vector and DEIM
        Need to embed conditional RHS calculation - WIP
        """
        q1_full = np.matmul(self.q1_V,q1_red)
        q1_full = np.reshape(q1_full,newshape=(Nx,Ny))

        q2_full = np.matmul(self.q2_V,q2_red)
        q2_full = np.reshape(q2_full,newshape=(Nx,Ny))

        q3_full = np.matmul(self.q3_V,q3_red)
        q3_full = np.reshape(q3_full,newshape=(Nx,Ny))

        q1nl, q2nl, q3nl = self.nonlinear_term_full(q1_full,q2_full,q3_full)

        q1nl_red = np.matmul(np.transpose(self.q1_V),q1nl.reshape(4096,))
        q2nl_red = np.matmul(np.transpose(self.q2_V),q2nl.reshape(4096,))
        q3nl_red = np.matmul(np.transpose(self.q3_V),q3nl.reshape(4096,))

        return -q1nl_red, -q2nl_red, -q3nl_red

    def nonlinear_term_full(self,q1,q2,q3):
        f1x, f1y, f2x, f2y, f3x, f3y = flux_reconstruction(q1,q2,q3) # these are all i+1/2

        # Periodicity
        pad = 1
        f1xtemp = periodic_bc(f1x,pad)
        f1ytemp = periodic_bc(f1y,pad)
        f2xtemp = periodic_bc(f2x,pad)
        f2ytemp = periodic_bc(f2y,pad)
        f3xtemp = periodic_bc(f3x,pad)
        f3ytemp = periodic_bc(f3y,pad)

        r1 = 1.0/dx*(f1xtemp[pad:Nx+pad,pad:Ny+pad]-f1xtemp[pad-1:Nx+pad-1,pad:Ny+pad]) + 1.0/dy*(f1ytemp[pad:Nx+pad,pad:Ny+pad]-f1ytemp[pad:Nx+pad,pad-1:Ny+pad-1])
        r2 = 1.0/dx*(f2xtemp[pad:Nx+pad,pad:Ny+pad]-f2xtemp[pad-1:Nx+pad-1,pad:Ny+pad]) + 1.0/dy*(f2ytemp[pad:Nx+pad,pad:Ny+pad]-f2ytemp[pad:Nx+pad,pad-1:Ny+pad-1])
        r3 = 1.0/dx*(f3xtemp[pad:Nx+pad,pad:Ny+pad]-f3xtemp[pad-1:Nx+pad-1,pad:Ny+pad]) + 1.0/dy*(f3ytemp[pad:Nx+pad,pad:Ny+pad]-f3ytemp[pad:Nx+pad,pad-1:Ny+pad-1])

        return r1, r2, r3