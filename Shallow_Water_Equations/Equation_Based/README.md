# Shallow-water equation solver

## Data generation
For generating data for our ML/POD-Galerkin projection work set `fvm_solve` in `parameters.py` to `True`. Make sure to set an appropriate value of `K` corresponding to the number of POD modes you want to retain for the Galerkin projection. This will generate a couple of data sets `snapshot_matrix_pod.py`, `snapshot_matrix_test.py` which will be used for both the POD and the convolutional autoencoders.

## POD Galerkin
For running a POD Galerkin reduced-order model test, change the `fvm_solve` parameter to `False` and make sure that the `gprom.generate_pod()` is **not** commented out. This call will compute the POD bases. Following this a POD-Galerkin test will be run through `gprom.load_pregenerated_pod()` and `gprom.solve()`. For redoing the POD-ROM you can comment out the `gprom.generate_pod()` function call.