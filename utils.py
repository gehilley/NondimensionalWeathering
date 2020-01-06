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

