filename = 'models/base_case.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import pickle as p

# Run model:
L_star = 7.61
X0_star = 5.27
Y0_star = 0.000342
v_star = 679
nx = 101
t_star = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, X0_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump((x, X, Y, L_star, X0_star, Y0_star, v_star, nx, t_star, dx_star), open(filename, 'wb'))
