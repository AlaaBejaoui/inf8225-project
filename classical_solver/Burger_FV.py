import numpy as np
import copy


class Burger_FV():

    def __init__(self, x_start=0, x_end=1, t_start=0, t_end=0.5, CFL=0.2, N=100, IC="Rarefaction", u_1eft=0, u_right=2):
        super(Burger_FV, self).__init__()
        self.x_start = x_start
        self.x_end = x_end
        self.t_start = t_start
        self.t_end = t_end
        self.CFL = CFL
        self.N = N
        self.IC = IC
        self.u_left = u_left
        self.u_right = u_right

        # Grid
        self.N = N
        self.dx = (x_end-x_start)/N
        # self.x = np.linspace(self.x_start+self.dx/2, self.x_end-self.dx/2, N)
        self.x = np.linspace(self.x_start-self.dx, self.x_end+self.dx, N+1)
        self.N_exact = 500
        self.x_exact = np.linspace(self.x_start, self.x_end, self.N_exact)


        # Initialization
        self.u = np.zeros((N+1, 1))
        self.u_old = np.zeros_like(u)
        self.u_exact = np.zeros((N_exact, 1))

        # Initial condition
        if (self.IC == "Rarefaction") or (self.IC == "Shock"):
            self.u_old[self.x<0.5] = self.u_left
            self.u_exact[self.x<0.5] = self.u_left
            self.u_old[self.x>=0.5] = self.u_right
            self.u_exact[self.x>=0.5] = self.u_right
        elif self.IC == "Sine":
            self.u_old = np.sin(self.x)
            self.u_exact = np.sin(self.x)
        else:
            raise NotImplementedError("Initial condition not implemented")

        # Initialze lists to store the results
        self.t_list = [self.t_start]
        self.u_list = [self.u_old]
        self.u_exact_list = [self.u_exact]


        
    def gudonovFlux(u_left, u_right):
        # Shock
        if u_left > u_right:
            # Compute the shock speed from the Rankine-Hugoniot condition
            s = (u_left+u_right)/2
            if s > 0:
                u = u_left
            else:
                u = u_right
        # Rarefaction
        elif u_left < u_right:
            a_left = u_left
            a_right = u_right
            if a_left > 0:
                u = u_left
            elif a_right < 0:
                u = u_right
            else:
                u = 0
        # Constant state: u_left = u_right
        else:
            u = u_left
        
        f = 0.5*u**2
        return f


    def exactSolution(u_left, u_right, x, t):
        # Shock
        if u_left > u_right:
            # Compute the shock speed from the Rankine-Hugoniot condition
            s = (u_left + u_right)/2
            if s*t > (x-0.5):
                u = u_left
            else:
                u = u_right
        # Rarefaction
        elif u_left < u_right:
            a_left = u_left
            a_right = u_right
            if a_left*t > (x-0.5):
                u = u_left
            elif a_right*t < (x-0.5):
                u = u_right
            else:
                u = (x-0.5)/t
        # Constant state: u_left = u_right
        else:
            u = u_left
        return u


    def solve(self):
        while t <= t_1:
            for i in range(Nexact):
                u_exact[i] = exactBurger(u_1, u_2, x_exact[i], t)
                u_exact_list = np.column_stack((u_exact_list, u_exact))
        
            # Compute the time step from the CFL condition
            # a_max = np.max(np.abs(u))
            # dt = (CFL*dx)/a_max
            dt = 0.001
    
            # if t+dt > t_1:
            #    dt = t_1-t
            # if t >= t_1:
            #    break 
    
            t += dt
            t_list.append(np.round(t, 2))
    
            u_old[0] = u_1
            u_old[-1] = u_old[-2]

            for i in range(1, Ncell):
                # Left boundary
                # if i == 0:
                #    u_l = u_1 # u[-1] 
                #    u_r = u[1]
                # Right boundary
                # elif i == Ncell-1:
                #    u_l = u[-2]
                #    u_r = u[-1] # u[0] 
                # Inner points
                # else:
                #    u_l = u[i-1]
                #    u_r = u[i+1]

                # Flux
                f_l = godFlux(u_old[i-1], u_old[i])    
                f_r = godFlux(u_old[i], u_old[i+1])

                # Update
                u[i] = u_old[i] - (dt/dx)*(f_r-f_l)

            u_old = copy.deepcopy(u)
            u_list = np.column_stack((u_list, u))

