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

def plot_models(filename, out_prefix, save_plots = False, plot_symbols = None, plot_colors = None, t_indexes = None):

    import pickle as p
    import matplotlib.pylab as plt
    import numpy as np

    (x, X, Y, L_star, Y0_star, v_star, nx, t_star, dx_star) = p.load(open(filename, 'rb'))

    plot_symbols = ['-' for i in range(len(t_star))] if plot_symbols is None else plot_symbols
    plot_indexes = t_indexes if t_indexes is not None else range(len(t_star))
    plot_colors = plot_colors if plot_colors is not None else ['r', 'b']

    plt.figure()
    plt.title(out_prefix)

    for (i, plot_symbol) in zip(plot_indexes, plot_symbols):
        plt.plot(x, X[i, :], plot_symbol, color=plot_colors[0], linewidth=1.0)
        plt.plot(x, Y[i, :], plot_symbol, color=plot_colors[1], linewidth=1.0)

<<<<<<< HEAD
        for (i, plot_symbol) in zip(plot_indexes, plot_symbols):
            plt.figure(1)
            plt.plot(X[i, :], -x, color+plot_symbol)
            plt.figure(2)
            plt.plot(Y[i, :], -x, color+plot_symbol)

    plt.figure(1)
    plt.axis([0, X0_star_max*1.1, -(np.max(x)), 0])
    plt.figure(2)
    plt.axis([0, 1.1, -(np.max(x)), 0])
=======
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel('$x^{*}$')
    plt.ylabel('$X^{*}$, $Y^{*}$')
    plt.axis([0, np.max(x), 0, 1.1])



    if save_plots:
        plt.savefig(out_prefix+'_chem.eps')
    else:
        plt.show()

def plot_cracking_models(filename, out_prefix, save_plots = False, plot_symbols = None, plot_colors = None, t_indexes = None, upper_L_crack = 7.4E-6, lower_L_crack = 1.5E-8):
    import pickle as p
    import matplotlib.pylab as plt
    import numpy as np

    import warnings

    warnings.filterwarnings("ignore")

    (x, X, Y, L_star, Y0_star, v_star, nx, t_star, dx_star) = p.load(open(filename, 'rb'))

    plot_symbols = ['-' for i in range(len(t_star))] if plot_symbols is None else plot_symbols
    plot_indexes = t_indexes if t_indexes is not None else range(len(t_star))

    plt.figure()
    plt.title(out_prefix)

    for (i, plot_symbol) in zip(plot_indexes, plot_symbols):
        L_crack = np.power(Y0_star,2) / np.power(1-X[i,:],2)
        plt.semilogy(x, L_crack, 'k' + plot_symbol, linewidth=1.0)
        plt.semilogy([0, max(x)], [upper_L_crack, upper_L_crack], 'k--', linewidth=1.0)
        plt.semilogy([0, max(x)], [lower_L_crack, lower_L_crack], 'k:', linewidth=1.0)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel('$x^{*}$')
    plt.ylabel('$L^{*}$')
    plt.axis([0, np.max(x), 1E-8, 1E-4])

>>>>>>> 8b59a00d59c026e1b3a3b8f7c3c6fe51a272c21a
    if save_plots:
        plt.savefig(out_prefix + '_crack.eps')
    else:
        plt.show()
