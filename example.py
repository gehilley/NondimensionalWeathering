from weathering_model import run_weathering_model
import numpy as np
import matplotlib.pylab as plt

# Run model:
L_star = 50
X0_star = 1
Y0_star = 1
v_star = 1
nx = 1000
t_star = np.array([0, 2, 4, 6, 8, 10, 12])
dx_star = L_star / float(nx)

x, X, Y = run_weathering_model(L_star, X0_star, v_star, Y0_star, t_star, dx_star=dx_star)

# X[0,:] is the first t_star, X[1,:] is the second t_star, etc

plt.ion()

for i in range(len(t_star)):
    plt.figure(1)
    plt.plot(x,X[i,:],'-')
    plt.figure(2)
    plt.plot(x,Y[i,:],'-')
