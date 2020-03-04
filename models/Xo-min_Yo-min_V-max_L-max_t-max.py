filename = 'models/Xo-min_Yo-min_V-max_L-max_t-max.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import pickle as p

# Run model:
L_star = 1900
X0_star = 0.122
Y0_star = 0.000171
v_star = 1040000000
nx = 101
t_star = np.array([0, 923, 1846, 2769, 3692, 4615, 5538, 6461, 7384, 8307, 9230])
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, X0_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump((x, X, Y, L_star, X0_star, Y0_star, v_star, nx, t_star, dx_star), open(filename, 'wb'))
