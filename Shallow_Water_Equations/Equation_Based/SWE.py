import numpy as np
np.random.seed(10)
from problem import shallow_water, shallow_water_rom, plot_coefficients
from parameters import fvm_solve, num_samples, num_train, num_steps_per_plot, ft, dt
from pyDOE import *

if __name__ == '__main__':
    if fvm_solve:
        print('Running multiple IC non-linear SWE')

        locs = np.array(-0.5+lhs(2,num_samples,criterion='center'))
        np.save('Locations.npy',locs) # For training

        for loc_num in range(num_samples):

            new_run = shallow_water(locs[loc_num])
            new_run.solve()
            
            if loc_num == 0:
                snapshot_matrix_pod = np.transpose(np.array(new_run.snapshots_pod))
            else: 
                temp = np.transpose(np.array(new_run.snapshots_pod))
                snapshot_matrix_pod = np.concatenate((snapshot_matrix_pod,temp),axis=-1)

        print('Saving training snapshot data for multiple runs')
        np.save('snapshot_matrix_pod.npy',snapshot_matrix_pod[:,:(num_train*Nt/num_steps_per_plot)])
        np.save('snapshot_matrix_test.npy',snapshot_matrix_pod[:,(num_train*Nt/num_steps_per_plot):])

    else:
        # Loading snapshots
        snapshot_matrix_pod = np.load('snapshot_matrix_pod.npy')
        snapshot_matrix_test = np.load('snapshot_matrix_test.npy')
        # Shape
        print('Shape of the train snapshot matrix:',np.shape(snapshot_matrix_pod))
        print('Shape of the test snapshot matrix:',np.shape(snapshot_matrix_test))
        
        # Initialize ROM class
        gprom = shallow_water_rom(snapshot_matrix_pod,snapshot_matrix_test)
        # # Compute POD coefficients (if not done previously)
        gprom.generate_pod()
        # Load POD coefficients
        gprom.load_pregenerated_pod()    
        # Do GP solve using equations
        gprom.solve()



    