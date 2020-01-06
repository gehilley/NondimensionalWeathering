

def test_weathering_model():
    from utils import pack_values
    from muscl import muscl, vanAlbada
    import numpy as np
    nx = 1000
    dx = 0.05
    Xo = 1
    x0 = np.zeros((nx,2))
    x0[:,0] = Xo
    t = np.array([0, 2, 4, 6, 8, 10, 12])*5
    vstar = 1
    r = 4
    Yostar = 1

    def bc_function(x):
        yb = x[-1,1]
        return (np.array([[Xo, 1],[0, 0]]), np.array([[Xo, yb],[0, 0]]))

    def flux_function(x):

        v = np.zeros_like(x)
        v[:,1] = vstar
        return x * v

    def source_function(x):
        X = x[:,0]
        Y = x[:,1]
        s = np.zeros_like(x)
        s[:,0] = -np.power(Y,r)*np.power(X,2)
        s[:,1] = -r*np.power(X,2)*np.power(Y,r)/Yostar
        return s

    def diffusion_function(x):
        x_d = np.zeros((x0.shape[0]+2,2))
        x_d[1:-1,:] = x
        (topBC, bottomBC) = bc_function(x)
        x_d[0,1] = topBC[0,1]
        x_d[-1,1] = bottomBC[0,1]
        q = np.diff(x_d[:,1]) / dx
        dxdt = np.zeros_like(x)
        dxdt[:,1] = vstar*np.diff(q) / dx
        return dxdt

    def prop_speed(x):

        v = np.zeros_like(x)
        v[:,1] = vstar
        return np.abs(v)

    def to_integrate(t, x):
        x_unpacked = pack_values(x, packing_geometry=x0.shape)
        dxdt_flux = muscl(x_unpacked, dx, bc_function, flux_function, vanAlbada, prop_speed = prop_speed, reconstruction='parabolic')
        dxdt_source = source_function(x_unpacked)
        dxdt_diffusion = diffusion_function(x_unpacked)
        dxdt = dxdt_flux + dxdt_source + dxdt_diffusion
        return pack_values(dxdt, packing_geometry=None)

    from scipy.integrate import solve_ivp
    out = solve_ivp(to_integrate, (np.min(t), np.max(t)), pack_values(x0, packing_geometry=None), method='LSODA', t_eval=t)
    y = out.y.T
    import matplotlib.pylab as plt

    plt.ion()

    x_a = np.arange(0,nx*dx,dx)
    for i in range(len(t)):
        this_y = pack_values(y[i], packing_geometry=x0.shape)
        plt.figure(1)
        plt.plot(x_a,this_y[:,0],'.')
        plt.figure(2)
        plt.plot(x_a,this_y[:,1],'.')