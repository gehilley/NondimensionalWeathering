def pack_values(values, packing_geometry=None):
    """
    pack_values: packs and unpacks values into vectors suitable for ODE integrators:

    Parameters:
    -----------
    values : ndarray of values to unpack / pack.  Array is n x m if values are to be packed.
    packing_geometry : A tuple of the output size of the packed values.  If packing_geometry is None, values will be
                        packed into an ((n*m) x 1) vector.  Values will be repacked row-wise (Fortran order).
    """

    import numpy as np
    packing_geometry = (np.prod(values.shape),) if packing_geometry is None else packing_geometry
    assert(np.prod(values.shape) == np.prod(np.array(packing_geometry)))
    return np.reshape(values, packing_geometry, order='F')

def plot_models(filename):

    import pickle as p
    import matplotlib.pylab as plt

    (x, X, Y, L_star, X0_star, Y0_star, v_star, nx, t_star, dx_star) = p.load(open(filename, 'rb'))

    # X[0,:] is the first t_star, X[1,:] is the second t_star, etc

    if(x.shape[0] == X.shape[1]):
        x_plot = x
    else:
        x_plot = x[0:-1]

    for i in range(len(t_star)):
        plt.figure(1)
        plt.plot(x_plot, X[i, :], '-')
        plt.figure(2)
        plt.plot(x_plot, Y[i, :], '-')

    plt.show()