filename = 'models/FinalRound_2022/YoBC_V2xBC.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import pickle as p

# Run model:

L_star = 9.51
Y0_star = 3.56E-04
v_star = 5.70E+02
nx = 101
t_star_max = 36.2

t_star = np.linspace(0,t_star_max,num=11)
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump((x, X, Y, L_star, Y0_star, v_star, nx, t_star, dx_star), open(filename, 'wb'))
