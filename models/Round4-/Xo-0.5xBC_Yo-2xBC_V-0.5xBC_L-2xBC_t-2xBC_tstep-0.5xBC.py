filename = 'models/Xo-0.5xBC_Yo-2xBC_V-0.5xBC_L-2xBC_t-2xBC_tstep-0.5xBC.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import pickle as p

# Run model:
L_star = 15.2
X0_star = 2.63
Y0_star = 0.000685
v_star = 340
nx = 101
t_star = np.array([0, 0.97, 1.94, 2.92, 3.89, 4.86, 5.83, 6.80, 7.78, 8.75, 9.72, 10.69, 11.66, 12.64, 13.61, 14.58, 15.55, 16.52, 17.50, 18.47, 19.44, 20.41, 21.38, 22.36, 23.33, 24.30, 25.27, 26.24, 27.22, 28.19, 29.16, 30.13, 31.10, 32.08, 33.05, 34.02, 34.99, 35.96, 36.94, 37.91, 38.88])
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, X0_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump((x, X, Y, L_star, X0_star, Y0_star, v_star, nx, t_star, dx_star), open(filename, 'wb'))
