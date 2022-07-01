tout = 0

def run_weathering_model(Lstar, vstar, Yostar, tstar, dxstar = 0.05, r = 0.25, method = 'LSODA', use_cfl = True):
    from weathering_model.utils import pack_values
    from weathering_model.muscl import muscl, vanAlbada
    import numpy as np

    nxstar = int(np.ceil(Lstar / dxstar))
    x0 = np.zeros((nxstar,2))
    x0[:,0] = 1.
    x0[:,1] = 0.0

    def bc_function(x):
        yb = x[-1, 1]
        return (np.array([[1, 1], [1, 1]]), np.array([[1, yb], [1, yb]]))

    def flux_function(x):

        v = np.zeros_like(x)
        v[:,1] = vstar
        return x * v

    def source_function(x):
        X = np.logical_and(x[:,0] > 0, x[:,0] <= 1) * x[:,0] + (x[:,0] > 1) * 1
        Y = np.logical_and(x[:,1] > 0, x[:,1] <= 1) * x[:,1] + (x[:,1] > 1)*1
        s = np.zeros_like(x)

        s[:,0] = -np.power(Y,r)*np.power(X,2)*(X > 0).astype(X.dtype)*(Y > 0).astype(Y.dtype)*(X <= 1).astype(X.dtype)*(Y <= 1.0).astype(Y.dtype)
        s[:,1] = -r*np.power(X,2)*np.power(Y,r)/Yostar*(X > 0).astype(X.dtype)*(Y > 0).astype(Y.dtype)*(X <= 1).astype(X.dtype)*(Y <= 1.0).astype(Y.dtype)
        return s

    def diffusion_function(x):
        x_d = np.zeros((x0.shape[0]+2,2))
        x_d[1:-1,1] = x[:,1]
        (topBC, bottomBC) = bc_function(x)
        x_d[0,1] = topBC[0,1]
        x_d[-1,1] = bottomBC[0,1]
        q = np.diff(x_d[:,1]) / dxstar
        dxdt = np.zeros_like(x)
        dxdt[:,1] = vstar*np.diff(q) / dxstar
        return dxdt

    def prop_speed(x):

        v = np.zeros_like(x)
        v[:,1] = vstar
        return np.abs(v)

    def to_integrate(t, x):
        x_unpacked = pack_values(x, packing_geometry=x0.shape)
        dxdt_flux = muscl(x_unpacked, dxstar, bc_function, flux_function, vanAlbada, prop_speed = prop_speed, reconstruction='parabolic')
        dxdt_flux[:,0] = 0
        dxdt_source = source_function(x_unpacked)
        dxdt_diffusion = diffusion_function(x_unpacked)
        dxdt_diffusion[:,0] = 0
        dxdt = dxdt_flux + dxdt_source + dxdt_diffusion
        global tout
        if tout < t:
            print(tout, flush = True)
            tout += 0.1

        return pack_values(dxdt, packing_geometry=None)

    from scipy.integrate import solve_ivp
    cfl_dt = 0.25 * dxstar / np.abs(vstar) if use_cfl else np.inf

    out = solve_ivp(to_integrate, (np.min(tstar), np.max(tstar)), pack_values(x0, packing_geometry=None), method=method, max_step = cfl_dt, t_eval=tstar, lband = 2, uband = 2)
    if out is None:
        print('problem')
    print(out)
    X_star = np.zeros((len(tstar),x0.shape[0]))
    Y_star = np.zeros((len(tstar),x0.shape[0]))

    this_y = out.y.T
    for i in range(len(tstar)):
        X_star[i,:] = pack_values(this_y[i,:],packing_geometry=x0.shape)[:,0]
        Y_star[i,:] = pack_values(this_y[i,:],packing_geometry=x0.shape)[:,1]
    return np.arange(0, Lstar, dxstar), X_star, Y_star

def test_weathering_model():
    from weathering_model.utils import pack_values
    from weathering_model.muscl import muscl, vanAlbada
    import numpy as np
    nx = 1000
    dx = 0.05
    Xo = 1
    x0 = np.zeros((nx,2))
    x0[:,0] = Xo
    t = np.array([0, 2, 4, 6, 8, 10, 12])
    vstar = 1
    r = 0.25
    Yostar = 1

    def bc_function(x):

        return (np.array([[1.0, 1],[0, 0]]), np.array([[1.0, yb],[0, 0]]))

    def flux_function(x):

        v = np.zeros_like(x)
        v[:,1] = vstar
        return x * v

    def source_function(x):
        X = x[:,0]
        Y = (x[:,1] > 0) * x[:,1]
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
        print(t)
        return pack_values(dxdt, packing_geometry=None)

    from scipy.integrate import solve_ivp
    out = solve_ivp(to_integrate, (np.min(t), np.max(t)), pack_values(x0, packing_geometry=None), method='RK45', t_eval=t)
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