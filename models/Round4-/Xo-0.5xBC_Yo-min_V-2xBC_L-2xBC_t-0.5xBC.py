filename = 'models/Xo-0.5xBC_Yo-min_V-2xBC_L-2xBC_t-0.5xBC.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import pickle as p

# Run model:
L_star = 15.2
X0_star = 2.63
Y0_star = 0.000171
v_star = 1360
nx = 101
t_star = np.array([0, 0.97, 1.94, 2.92, 3.89, 4.86, 5.83, 6.80, 7.78, 8.75, 9.72])
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, X0_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump((x, X, Y, L_star, X0_star, Y0_star, v_star, nx, t_star, dx_star), open(filename, 'wb'))
