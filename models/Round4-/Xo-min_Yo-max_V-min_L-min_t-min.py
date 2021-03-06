filename = 'models/Xo-min_Yo-max_V-min_L-min_t-min.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import pickle as p

# Run model:
L_star = 0.0381
X0_star = 0.122
Y0_star = 0.00171
v_star = 0.255
nx = 101
t_star = np.array([0, 0.000017, 0.000035, 0.000052, 0.000070, 0.000087, 0.000104, 0.000122, 0.000139, 0.000157, 0.000174])
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, X0_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump((x, X, Y, L_star, X0_star, Y0_star, v_star, nx, t_star, dx_star), open(filename, 'wb'))
