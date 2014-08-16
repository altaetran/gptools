import numpy as np
import gptools

def test_gradient_inputs():
    # This test checks whether or not gradient inputs are accepted by gptools
    f_X = np.random.RandomState(0).randn(5,2)
    # List of function evaluations
    f_y = (f_X[:,0]**2 + f_X[:,1]**2).tolist()
    # Function evaluations
    f_y_0 = f_X[:,0]**2 + f_X[:,1]**2
    # list of Gradients
    g_y = (2*f_X).tolist()
    # Gradient components
    g_y_0 = 2*f_X[:,0]
    g_y_1 = 2*f_X[:,1]
    # List of hessians
    h_y = []
    # List of hessian components
    h_y_00 = []
    h_y_01 = []
    h_y_11 = []
    for k in range(len(f_y)):
        h_y.append(np.array([[2, 0],[0, 2]]))
        h_y_00.append(2)
        h_y_01.append(0)
        h_y_11.append(2)

    # List of errors
    err_y = f_y
    # Errors
    err_y_0 = f_y_0
    # Gradient error components
    err_g_0 = f_y_0
    err_g_1 = 2*f_y_0
    # List of gradient errors
    err_g_full = np.vstack((err_g_0, err_g_1)).T.tolist()
    
    n_dims = 2
    length_scales = np.random.lognormal(size=n_dims).tolist()
    K1 = gptools.SquaredExponentialKernel(num_dim=2,
        initial_params=[10] + length_scales)
    K2 = gptools.SquaredExponentialKernel(num_dim=2,
        initial_params=[10] + length_scales)

    gp1 = gptools.GaussianProcess(K1)
    gp2 = gptools.GaussianProcess(K2)


    # Input list of function observations, gradients, and hessians
    gp1.add_data_list(f_X, f_y, err_y=err_y)
    gp1.add_data_list(f_X, g_y, err_y=err_g_full, n=1)
    gp1.add_data_list(f_X, h_y, err_y=err_y, n=2)
    #Input funtion observations, gradients, and hessians. 
    gp2.add_data(f_X, f_y_0, err_y=err_y)
    gp2.add_data(f_X, g_y_0, err_y=err_g_0, n=np.vstack((np.ones(len(f_X)), np.zeros(len(f_X)))).T)
    gp2.add_data(f_X, g_y_1, err_y=err_g_1, n=np.vstack((np.zeros(len(f_X)), np.ones(len(f_X)))).T)
    gp2.add_data(f_X, h_y_00, err_y=err_y, n=np.vstack((2*np.ones(len(f_X)), np.zeros(len(f_X)))).T)
    gp2.add_data(f_X, h_y_01, err_y=err_y, n=np.vstack((np.ones(len(f_X)), np.ones(len(f_X)))).T)
    gp2.add_data(f_X, h_y_11, err_y=err_y, n=np.vstack((np.zeros(len(f_X)), 2*np.ones(len(f_X)))).T)

    k1 = gp1.compute_Kij(gp1.X, None, gp1.n, None)
    k2 = gp2.compute_Kij(gp1.X, None, gp1.n, None)

    print([gp1.predict([1,2])])
    print([gp2.predict([1,2])])

    np.testing.assert_array_almost_equal(k1, k2, decimal=8)

test_gradient_inputs()
