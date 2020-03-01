filename = 'models/base_case.p'

from weathering_model.weathering_model import run_weathering_model
import numpy as np
import matplotlib.pylab as plt
import pickle as p

# Run model:
L_star = 0.228
X0_star = 5.27
Y0_star = 0.000342
v_star = 2.07
nx = 101
t_star = np.array([0, 4.4, 8.8, 13.2, 17.6, 22, 26.4, 30.8, 35.2, 39.6, 44]) / 1000.0
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, X0_star, v_star, Y0_star, t_star, dxstar=dx_star)

p.dump([x, X, Y], open(filename,'wb'))

# X[0,:] is the first t_star, X[1,:] is the second t_star, etc

plt.ion()

for i in range(len(t_star)):
    plt.figure(1)
    plt.plot(x,X[i,:],'-')
    plt.figure(2)
    plt.plot(x,Y[i,:],'-')
