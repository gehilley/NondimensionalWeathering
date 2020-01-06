def test_packing():
    from utils import pack_values

    import numpy as np
    print('packing testing...')
    values_in = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]], dtype=float)
    values_packed = pack_values(values_in)
    assert(np.all(values_packed == np.array([1,3,5,7,9,2,4,6,8,10], dtype=float)))
    values_out = pack_values(values_packed,packing_geometry=(5,2))
    assert(np.all(values_out == values_in))
    print('OK.')

def test_single_square_wave():
    from utils import pack_values
    from muscl import muscl, superbee
    import numpy as np
    x0 = np.zeros((200,1))
    x0[20:60] = 1.0
    t = np.array([0, 10, 20, 30, 40, 50, 60])
    dx = 0.5
    v = -1.0

    def bc_function(x):
        return (np.array([[0],[0]]), np.array([[0],[0]]))

    def flux_function(x):

        return x * v

    def prop_speed(x):
        return np.ones_like(x)*v

    def to_integrate(t, x):
        return pack_values(muscl(pack_values(x, packing_geometry=x0.shape), dx, bc_function, flux_function, superbee, prop_speed = prop_speed, reconstruction='linear'), packing_geometry=None)

    from scipy.integrate import solve_ivp
    out = solve_ivp(to_integrate, (np.min(t), np.max(t)), pack_values(x0, packing_geometry=None), method='LSODA', t_eval=t)
    y = out.y.T
    import matplotlib.pylab as plt

    plt.ion()

    for i in range(len(t)):
        plt.plot(y[i],'.')

def test_two_square_waves():
    from utils import pack_values
    from muscl import muscl, minmod
    import numpy as np
    x0 = np.zeros((200,2))
    x0[20:60,0] = np.ones((40,))
    x0[160:,1] = np.ones((40,))
    t = np.array([0, 10, 20, 30, 40, 50, 60])
    dx = 0.5
    v = np.array([[1.0, -1.0]])

    def bc_function(x):
        return (np.array([[0],[0]]), np.array([[0],[0]]))

    def flux_function(x):

        import numpy.matlib as matlib
        return x * matlib.repmat(v,x.shape[0],1)

    def prop_speed(x):

        import numpy.matlib as matlib
        return matlib.repmat(v,x.shape[0],1)

    def to_integrate(t, x):
        return pack_values(muscl(pack_values(x, packing_geometry=x0.shape), dx, bc_function, flux_function, minmod, prop_speed = prop_speed, reconstruction='linear'), packing_geometry=None)

    from scipy.integrate import solve_ivp
    out = solve_ivp(to_integrate, (np.min(t), np.max(t)), pack_values(x0, packing_geometry=None), method='LSODA', t_eval=t)
    y = out.y.T
    import matplotlib.pylab as plt

    plt.ion()

    for i in range(len(t)):
        this_y = pack_values(y[i], packing_geometry=x0.shape)
        plt.figure(1)
        plt.plot(this_y[:,0],'.')
        plt.figure(2)
        plt.plot(this_y[:,1],'.')