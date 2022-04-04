filename = 'models/Xo-max_Yo-max_V-min_L-min_t-max.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import pickle as p

# Run model:
L_star = 0.0381
X0_star = 10.4
Y0_star = 0.00171
v_star = 0.255
nx = 101
t_star = np.array([0, 923, 1846, 2769, 3692, 4615, 5538, 6461, 7384, 8307, 9230])
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, X0_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump((x, X, Y, L_star, X0_star, Y0_star, v_star, nx, t_star, dx_star), open(filename, 'wb'))
