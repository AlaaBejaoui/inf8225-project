import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return 300 - 80*np.exp(-0.5*(x/0.3)**2)


#Parameters
x_start = -1
x_end = 1
y_start = -1
y_end = 1
nx = 21
ny = 21
lambda_ = 0.0262
rho = 1.293
c = 1005

#Temperaturleitzahl
a=lambda_/(rho*c)

#Boundary condition
T_d = 400

#Grid 
dx = (x_end-x_start)/(nx-1)
dy = (y_end-y_start)/(ny-1)
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)

#Intialize ...
A = np.zeros((nx*ny, nx*ny))
r = np.zeros((nx*ny, 1))


for j in range(ny):
    for i in range(nx):
        i_0 = i+j*nx
        i_l = i_0-1
        i_r = i_0+1
        i_u = i_0+nx
        i_d = i_0-nx
          
        #Inner points
        if (0 < i) and (i < nx-1) and  (0 < j) and (j < ny-1):
            A[i_0, i_0] = -2*a/dx**2-2*a/dy**2
            A[i_0, i_u] = a/dy**2
            A[i_0, i_d] = a/dy**2
            A[i_0, i_r] = a/dx**2
            A[i_0, i_l] = a/dx**2
            r[i_0] = 0
        #Boundary
        else:
            #Upper boundary
            if j == ny-1:
                A[i_0, i_0] = 1
                r[i_0] = f(x[i]) 
            #Lower boundary
            if j == 0:
                A[i_0, i_0] = 1
                r[i_0] = T_d
            #Left and right boundary
            if (0 < j) and (j < nx-1): 
                if i == 0: #linker Rand, adiabat
                    A[i_0, i_0] = 1
                    A[i_0, i_r] = -1
                if i == nx-1: #rechter Rand, adiabat
                    A[i_0, i_0] = 1
                    A[i_0, i_l] = -1
             
# Solve LGS 
T_h = np.linalg.solve(A, r)

#Plot
T_h = np.reshape(T_h, (nx, ny))
fig, ax = plt.subplots()
contourPlot = ax.contourf(x, y, T_h, levels = 20, cmap = "coolwarm")
ax.contour(contourPlot, linewidths = 1, colors = "k")
ax.set_title('Title')
ax.set_xlabel("x")
ax.set_ylabel("y")
colorBar = plt.colorbar(contourPlot)
colorBar.set_label('T[K]')
plt.show()

