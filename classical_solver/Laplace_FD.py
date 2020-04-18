import numpy as np

class LaplaceFD():

    def __init__(self, x_start=-1, x_end=1, y_start=-1, y_end=1, nx=21, ny=21, lambda_=0.0262, rho=1.293, c=1005, 
    BC_left="Neumann", BC_right="Neumann", BC_down="Dirichlet", T_down=400, BC_up="Dirichlet", T_up=400):
        
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.nx = nx
        self.ny = ny
        self.lambda_ = lambda_
        self.rho = rho
        self.c = c
        self.T = np.zeros((nx*ny,1))
        # self.BC_left = BC_left
        # self.BC_right = BC_right
        # self.BC_down = BC_down
        # self.BC_up = BC_up

        @property
        def dx(self):
            return (self.x_end-self.x_start)/(self.nx-1)

        @property
        def dy(self):
            return (self.y_end-self.y_start)/(self.ny-1)

        @property
        def dx(self):
            return (self.x_end-self.x_start)/(self.nx-1)

        @property
        def dx(self):
            return (self.x_end-self.x_start)/(self.nx-1)

        
        # Compute the thermal diffusivity
        @property
        def a(self):
            return self.lambda_/(self.rho*self.c)

        # TODO: set boundary condition
        # Boundary conditions
        def set_boundary(self):
            if BC_down == "Dirichlet":
                self.T_down = T_down
            else:
                raise NotImplementedError("Boundary condition not implemented")
            if BC_up == "Dirichlet":
                self.T_up == T_up
            else:
                raise NotImplementedError("Boundary condition not implemented")
            if BC_left == "Neumann":
                pass
            else:
                raise NotImplementedError("Boundary condition not implemented")
            if BC_right == "Neumann":
                pass
            else:
                raise NotImplementedError("Boundary condition not implemented")

        self.x = np.linspace(self.x_start, self.x_end, self.nx)
        self.y = np.linspace(self.y_start, self.y_end, self.ny)

    # def f(x):
    #  return 300 - 80*np.exp(-0.5*(x/0.3)**2)

    def solve(self):
        
        self.set_boundary()
        # Initialization
        A = np.zeros((self.nx*self.ny, self.nx*self.ny))
        r = np.zeros((self.nx*self.ny, 1))
        
        for j in range(self.ny):
            for i in range(self.nx):
                i_0 = i+j*self.nx
                i_left = i_0-1
                i_right = i_0+1
                i_up = i_0+self.nx
                i_down = i_0-self.nx
                
                # Inner points
                if (0 < i) and (i < self.nx-1) and  (0 < j) and (j < self.ny-1):
                    A[i_0, i_0] = -2*self.a/self.dx**2-2*self.a/self.dy**2
                    A[i_0, i_up] = self.a/self.dy**2
                    A[i_0, i_down] = self.a/self.dy**2
                    A[i_0, i_right] = self.a/self.dx**2
                    A[i_0, i_left] = self.a/self.dx**2
                    r[i_0] = 0

                # Boundary
                else:
                    # Upper boundary
                    if j == self.ny-1:
                        A[i_0, i_0] = 1
                        # r[i_0] = f(x[i]) 
                        r[i_0] = self.T_up
                    
                    # Lower boundary
                    if j == 0:
                        A[i_0, i_0] = 1
                        r[i_0] = self.T_down

                    # Left and right boundary
                    if (0 < j) and (j < self.nx-1): 
                        # Left boundary
                        if i == 0:
                            A[i_0, i_0] = 1
                            A[i_0, i_right] = -1
                        # Right boundary
                        if i == self.nx-1: 
                            A[i_0, i_0] = 1
                            A[i_0, i_l] = -1
             
        # Solve the system of linear equations 
        self.T = np.linalg.solve(A, r)

    
    def plot_results(self):
        import matplotlib.pyplot as plt

        T_plot = np.reshape(self.T, (self.nx, self.ny))
        fig, ax = plt.subplots()
        contourPlot = ax.contourf(self.x, self.y, T_plot, levels=20, cmap="coolwarm")
        ax.contour(contourPlot, linewidths=1, colors="k")
        ax.set_title("Title")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        colorBar = plt.colorbar(contourPlot)
        colorBar.set_label("T[K]")
        plt.show()

        
    