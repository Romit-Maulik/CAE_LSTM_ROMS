import numpy as np

def galerkin_projection(phi_trunc,cf_trunc,sm_mean,tsteps,Rnum,dt,dx,num_modes):
    # Setup offline operators
    # Function for linear operator

    dataset_temp = np.copy(cf_trunc) 
    
    def linear_operator(u):# Requires ghost-points - this is laplacian
        u_per = np.zeros(shape=(np.shape(u)[0]+2),dtype='double')
        u_per[1:-1] = u[:]
        u_per[0] = u[-1]
        u_per[-1] = u[0]

        # ulinear = np.zeros(shape=(np.shape(u)[0]),dtype='double')
        ulinear = (u_per[0:-2] + u_per[2:] - 2.0*u_per[1:-1])/(dx*dx)

        return ulinear

    def nonlinear_operator(u,g):

        g_per = np.zeros(shape=(np.shape(g)[0]+2),dtype='double')
        g_per[1:-1] = g[:]
        g_per[0] = g[-1]
        g_per[-1] = g[0]

        dgdx = (g_per[0:-2] - g_per[2:])/(2.0*dx)

        return u*dgdx


    # Calculate mode-wise offline operators
    lin_ubar = 1.0/Rnum*linear_operator(sm_mean)
    nlin_ubar = nonlinear_operator(sm_mean,sm_mean)


    # Calculate linear and non-linear terms
    b1k = np.zeros(shape=(np.shape(cf_trunc)[0]),dtype='double')
    b2k = np.zeros(shape=(np.shape(cf_trunc)[0]),dtype='double')    
    lik_1 = np.zeros(shape=(np.shape(cf_trunc)[0],np.shape(cf_trunc)[0]),dtype='double')    
    lik_2 = np.zeros(shape=(np.shape(cf_trunc)[0],np.shape(cf_trunc)[0]),dtype='double')    
    nijk = np.zeros(shape=(np.shape(cf_trunc)[0],np.shape(cf_trunc)[0],np.shape(cf_trunc)[0]),dtype='double')    
    
    for k in range(num_modes):
        b1k[k] = np.sum(lin_ubar[:]*phi_trunc[:,k])
        b2k[k] = np.sum(nlin_ubar[:]*phi_trunc[:,k])

        for i in range(num_modes):
            lin_phi = 1.0/Rnum*linear_operator(phi_trunc[:,i])
            lik_1[i,k] = np.sum(lin_phi[:]*phi_trunc[:,k])

            nlin_phi = nonlinear_operator(sm_mean,phi_trunc[:,i]) + nonlinear_operator(phi_trunc[:,i],sm_mean)
            lik_2[i,k] = np.sum(nlin_phi[:]*phi_trunc[:,k])

            for j in range(num_modes):
                nlin_phi = nonlinear_operator(phi_trunc[:,i],phi_trunc[:,j])
                nijk[i,j,k] = np.sum(nlin_phi[:]*phi_trunc[:,k])

    # Operators fixed - one time cost
    # Evaluation using GP

    def gp_rhs(b1k,b2k,lik_1,lik_2,nijk,state):
        rhs = np.zeros(np.shape(state)[0])

        rhs = b1k[:] + b2k[:] 
        rhs = rhs + np.matmul(state,lik_1) + np.matmul(state,lik_2)

        # Nonlinear global operator
        for k in range(num_modes):
            rhs[k] = rhs[k] + np.matmul(np.matmul(nijk[:,:,k],state),state)

        return rhs

    # gp_rhs(b1k,b2k,lik_1,lik_2,nijk,cf_trunc[:,0])
    state = dataset_temp[:,0]
    state_tracker = np.zeros(shape=(np.shape(tsteps)[0],np.shape(state)[0]),dtype='double')

    trange = np.arange(int(np.shape(tsteps)[0])-1)
    for t in trange:
        state_tracker[t,:] = state[:]
        # TVDRK3 - POD GP
        rhs = gp_rhs(b1k,b2k,lik_1,lik_2,nijk,state)
        l1 = state + dt*rhs

        rhs = gp_rhs(b1k,b2k,lik_1,lik_2,nijk,l1)

        l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs

        rhs = gp_rhs(b1k,b2k,lik_1,lik_2,nijk,l2)

        state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:]

    return state, np.transpose(state_tracker)