filename = 'models/Round5/YoBC_V0.5xBC.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import pickle as p

# Run model:
L_star = 15.3
Y0_star = 7.83E-04
v_star = 2.92E+02
nx = 101
t_star_max = 45.3
t_star = np.linspace(0,t_star_max,num=11)
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump((x, X, Y, L_star, Y0_star, v_star, nx, t_star, dx_star), open(filename, 'wb'))
