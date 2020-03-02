filename = 'models/Xo-max_Yo-max_V-max_L-min_t-min.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import pickle as p

# Run model:
L_star = 0.00114
X0_star = 10.4
Y0_star = 0.00171
v_star = 966000
nx = 101
t_star = np.array([0, 0.000059, 0.000118, 0.000177, 0.000236, 0.000296, 0.000355, 0.000414, 0.000473, 0.000532, 0.000591])
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, X0_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump((x, X, Y, L_star, X0_star, Y0_star, v_star, nx, t_star, dx_star), open(filename, 'wb'))
