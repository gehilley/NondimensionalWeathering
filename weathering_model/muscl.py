def muscl(x, dx, bc_function, flux_function, flux_limiter, prop_speed = None, reconstruction = 'linear'):
    """
    Computes and returns the history of x0.

    Parameters
    ----------
    x : numpy.ndarray
        The  value of quantity
        as an m x n array of floats.
    dx : float
        The distance between two consecutive locations.
    bc_function : callable that accepts x and returns dxdt at
        boundaries.  First tuple is top (x, dx/dy), bottom
        tuple is bottom (x, dx/dy).
    flux_function : callable that accepts x and returns
        flux.
    flux_limiter : callable that accepts x and returns
        limited slope.
    prop_speed : callable that
    reconstruction : flag that specifies order of muscl reconstruction
        (only 'linear' is implemented at this time).

    Returns
    -------
    dxdt : list of numpy.ndarray objects
        The rate of change of x, suitable for ode integration
        .
    """

    import numpy as np
    geometry = x.shape

    if reconstruction == 'linear':
        # Add boundary conditions:
        x_c = np.zeros((geometry[0] + 3, geometry[1]))
        x_c[1:-2][:] = x[:][:]
        (topBCs, bottomBCs) = bc_function(x)
        x_c[0][:] = topBCs[0]
        x_c[-2][:] = bottomBCs[0]
        x_c[-1][:] = bottomBCs[0] + bottomBCs[1]*dx
        # Compute the limited slope.
        sigma = flux_limiter(x_c)
        # Reconstruct values at cell boundaries.
        x_centered = x_c[1:-2][:]
        x_plus1 = x_c[2:-1][:]
        x_plus2 = x_c[3:][:]
        x_minus1 = x_c[0:-3][:]
        sigma_centered = sigma[1:-2][:]
        sigma_minus1 = sigma[0:-3][:]
        sigma_plus1 = sigma[2:-1][:]
        x_l_ip12 = x_centered + 0.5*sigma_centered*(x_plus1-x_centered)
        x_r_ip12 = x_plus1 - 0.5*sigma_plus1*(x_plus2 - x_plus1)
        x_l_im12 = x_minus1 + 0.5*sigma_minus1*(x_centered-x_minus1)
        x_r_im12 = x_centered - 0.5*sigma_centered*(x_plus1-x_centered)
    elif reconstruction == 'parabolic':
        # Add boundary conditions:
        x_c = np.zeros((geometry[0] + 4, geometry[1]))
        x_c[2:-2][:] = x[:][:]
        (topBCs, bottomBCs) = bc_function(x)
        x_c[1][:] = topBCs[0]
        x_c[0][:] = topBCs[0] - topBCs[1]*dx
        x_c[-2][:] = bottomBCs[0]
        x_c[-1][:] = bottomBCs[0] + bottomBCs[1]*dx
        # Compute the limited slope.
        sigma = flux_limiter(x_c)
        sigma_centered = sigma[2:-2][:]
        sigma_minus1 = sigma[1:-3][:]
        sigma_plus1 = sigma[3:-1][:]
        # Reconstruct values at cell boundaries.
        x_centered = x_c[2:-2][:]
        x_minus1 = x_c[1:-3][:]
        x_minus2 = x_c[0:-4][:]
        x_plus1 = x_c[3:-1][:]
        x_plus2 = x_c[4:][:]
        dx_ip12 = x_plus1 - x_centered
        dx_ip32 = x_plus2 - x_plus1
        dx_im12 = x_centered - x_minus1
        dx_im32 = x_minus1 - x_minus2
        kappa = 1/3
        x_l_ip12 = x_centered + (sigma_centered / 4.0)*((1.0-kappa)*dx_im12 + (1.0+kappa)*dx_ip12)
        x_r_ip12 = x_plus1 - (sigma_plus1 / 4.0)*((1.0-kappa)*dx_ip32 + (1.0+kappa)*dx_ip12)
        x_l_im12 = x_minus1 + (sigma_minus1 / 4.0)*((1.0-kappa)*dx_im32 + (1.0+kappa)*dx_im12)
        x_r_im12 = x_centered - (sigma_centered / 4.0)*((1.0-kappa)*dx_ip12 + (1.0+kappa)*dx_im12)
    else:
        return

    # Compute the propagation speed:
    if prop_speed is None:
        # TODO: Calculate propagation speed numerically:
        pass

    else:
        a_ip12 = np.maximum(np.abs(prop_speed(x_l_ip12)),np.abs(prop_speed(x_r_ip12)))
        a_im12 = np.maximum(np.abs(prop_speed(x_l_im12)),np.abs(prop_speed(x_r_im12)))

    F_l_ip12 = flux_function(x_l_ip12)
    F_r_ip12 = flux_function(x_r_ip12)
    F_l_im12 = flux_function(x_l_im12)
    F_r_im12 = flux_function(x_r_im12)

    Fstar_im12 = 0.5*(F_r_im12 + F_l_im12 - a_im12*(x_r_im12 - x_l_im12))
    Fstar_ip12 = 0.5*(F_r_ip12 + F_l_ip12 - a_ip12*(x_r_ip12 - x_l_ip12))

    return -(Fstar_ip12 - Fstar_im12) / dx

def calc_r(e, epsilon = 1E-20):
    import numpy as np
    r = np.zeros_like(e)
    r[1:-1] = (e[1:-1][:] - e[0:-2][:] + epsilon) / (e[2:][:] - e[1:-1][:] + epsilon)
    return r

def minmod(e, epsilon = 1E-20):
    """
    Computes the minmod approximation of the slope.

    Parameters
    ----------
    e : list or numpy.ndarray
        The input values as a 1D array of floats.
    dx : float
        The grid-cell width.

    Returns
    -------
    sigma : numpy.ndarray
        The minmod-approximated slope
        as a 1D array of floats.
    """
    import numpy as np
    r = calc_r(e, epsilon=epsilon)

    return np.maximum(0.0,np.minimum(1.0,r))

def superbee(e, epsilon = 1E-20):
    """
    Computes the superbee approximation of the slope.

    Parameters
    ----------
    e : list or numpy.ndarray
        The input values as a 1D array of floats.
    dx : float
        The grid-cell width.

    Returns
    -------
    sigma : numpy.ndarray
        The minmod-approximated slope
        as a 1D array of floats.
    """
    import numpy as np
    r = calc_r(e, epsilon=epsilon)

    return np.maximum(0, np.minimum(2*r, 1), np.minimum(2.0, r))

def vanAlbada(e, epsilon = 1E-20):
    """
    Computes the superbee approximation of the slope.

    Parameters
    ----------
    e : list or numpy.ndarray
        The input values as a 1D array of floats.
    dx : float
        The grid-cell width.

    Returns
    -------
    sigma : numpy.ndarray
        The minmod-approximated slope
        as a 1D array of floats.
    """
    import numpy as np
    r = calc_r(e, epsilon=epsilon)

    return 2*r / (1+np.power(r,2))

