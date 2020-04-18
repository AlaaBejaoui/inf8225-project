import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.use("TkAgg")


def init_animation():
    fig = plt.figure()
    ax = plt.axes()
    ax.grid()
    ax.set_xlabel("x")
    ax.set_xlim((x_0, x_1))
    ax.set_ylabel("u")
    ax.set_ylim((0, 2))
    ax.set_title(f"t = {t_0}")
    numerical_solution, = ax.plot(x, u_old, "ro")
    exact_solution, = ax.plot(x_exact, u_exact, "k")
    ax.legend(["Numerical solution", "Exact solution"])
    return None


def animate(i):
    y = u_list[:, i]
    y_exact = u_exact_list[:, i]
    numerical_solution.set_data(x, y)
    exact_solution.set_data(x_exact, y_exact)
    ax.set_title(f"t = {t_list[i]}")
    return numerical_solution, exact_solution



# Plot
ani = animation.FuncAnimation(fig, animate, frames=np.arange(len(t_list)), interval=10)
plt.show()