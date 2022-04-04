filename = 'models/Xo-max_Yo-min_V-0.5xBC_L-2xBC_t-2xBC.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import pickle as p

# Run model:
L_star = 15.2
X0_star = 10.4
Y0_star = 0.000171
v_star = 340
nx = 1001
t_star = np.array([0, 3.89, 7.78, 11.67, 15.56, 19.45, 23.34, 27.23, 31.12, 35.01, 38.90])
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, X0_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump((x, X, Y, L_star, X0_star, Y0_star, v_star, nx, t_star, dx_star), open(filename, 'wb'))
