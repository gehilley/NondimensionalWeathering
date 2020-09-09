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

def plot_models(filenames, out_prefix, save_plots = False, plot_symbols = None, plot_colors = None, t_indexes = None):

    import pickle as p
    import matplotlib.pylab as plt
    import numpy as np

    X0_star_max = 0.0
    plot_symbols = ['-' for i in range(len(filenames))] if plot_symbols is None else plot_symbols
    plot_indexes = t_indexes if t_indexes is not None else range(len(t_star))
    pcs = ['' for i in range(len(plot_indexes))] if plot_colors is None else plot_colors

    for (filename, color) in zip(filenames, pcs):

        (x, X, Y, L_star, X0_star, Y0_star, v_star, nx, t_star, dx_star) = p.load(open(filename, 'rb'))
        X0_star_max = X0_star_max if X0_star_max > X0_star else X0_star

        for (i, plot_symbol) in zip(plot_indexes, plot_symbols):
            plt.figure(1)
            plt.plot(X[i, :], -x, color+plot_symbol)
            plt.figure(2)
            plt.plot(Y[i, :], -x, color+plot_symbol)

    plt.figure(1)
    plt.axis([0, X0_star_max*1.1, -(np.max(x)), 0])
    plt.figure(2)
    plt.axis([0, 1.1, -(np.max(x)), 0])
    if save_plots:
        plt.figure(1)
        plt.savefig(out_prefix+'_FeO.eps')
        plt.figure(2)
        plt.savefig(out_prefix+'_O2.eps')
    else:
        plt.show()
    plt.figure(1)
    plt.close()
    plt.figure(2)
    plt.close()