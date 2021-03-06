filename = 'models/Xo-min_Yo-min_V-1magU_L-min_t-1magU.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import pickle as p

# Run model:
L_star = 0.0381
X0_star = 0.122
Y0_star = 0.000171
v_star = 6790
nx = 101
t_star = np.array([0, 19.4, 38.8, 58.2, 77.6, 97.0, 116.4, 135.8, 155.2, 174.6, 194])
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, X0_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump((x, X, Y, L_star, X0_star, Y0_star, v_star, nx, t_star, dx_star), open(filename, 'wb'))
