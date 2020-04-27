import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append(os.getcwd())

def plot_results_hyperbolic(filename):

    with open(f"results/{filename}.pkl", "rb") as f:
        results_dict = pickle.load(f)

    x = results_dict["x"] 
    t = results_dict["t"] 
    nn_solution = results_dict["nn_solution"] 
    exact_solution = results_dict["exact_solution"] 
    error = results_dict["error"] 
    training_loss = results_dict["training_loss"]

    # Plot NN solution
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    contour_plot = ax.contourf(x, t, nn_solution, levels=40, cmap="coolwarm")
    ax.contour(contour_plot, linewidths=1, colors="k")
    plt.colorbar(contour_plot)
    plt.savefig(f"figures/{filename}_nn_solution.svg")

    # Plot exact solution
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    contour_plot = ax.contourf(x, t, exact_solution, levels=40, cmap="coolwarm")
    ax.contour(contour_plot, linewidths=1, colors="k")
    plt.colorbar(contour_plot)
    plt.savefig(f"figures/{filename}_exact_solution.svg")

    # Plot error
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    contour_plot = ax.contourf(x, t, error, levels=40, cmap="coolwarm")
    ax.contour(contour_plot, linewidths=1, colors="k")
    plt.colorbar(contour_plot)
    plt.savefig(f"figures/{filename}_error.svg")

    # Plot loss
    fig, ax = plt.subplots()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.semilogy(training_loss)
    plt.savefig(f"figures/{filename}_loss.svg")


def plot_results_elliptic(filename):

    with open(f"results/{filename}.pkl", "rb") as f:
        results_dict = pickle.load(f)

    X = results_dict["X"] 
    Y = results_dict["Y"] 
    nn_solution = results_dict["nn_solution"] 
    exact_solution = results_dict["exact_solution"] 
    error = results_dict["error"] 
    training_loss = results_dict["training_loss"]

    # Plot NN solution
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    contour_plot = ax.contourf(X, Y, nn_solution, levels=40, cmap="coolwarm")
    ax.contour(contour_plot, linewidths=1, colors="k")
    plt.colorbar(contour_plot)
    plt.savefig(f"figures/{filename}_nn_solution.svg")

    # Plot exact solution
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    contour_plot = ax.contourf(X, Y, exact_solution, levels=40, cmap="coolwarm")
    ax.contour(contour_plot, linewidths=1, colors="k")
    plt.colorbar(contour_plot)
    plt.savefig(f"figures/{filename}_exact_solution.svg")

    # Plot error
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    contour_plot = ax.contourf(X, Y, error, levels=40, cmap="coolwarm")
    ax.contour(contour_plot, linewidths=1, colors="k")
    plt.colorbar(contour_plot)
    plt.savefig(f"figures/{filename}_error.svg")

    # Plot loss
    fig, ax = plt.subplots()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.semilogy(training_loss)
    plt.savefig(f"figures/{filename}_loss.svg")

def plot_results_FD(filename):

    with open(f"results/{filename}.pkl", "rb") as f:
        results_dict = pickle.load(f)

    X = results_dict["X"]
    Y = results_dict["Y"]
    FD_solution = results_dict["FD_solution"]
    error = results_dict["error"]

    # Plot FD_solution
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    contour_plot = ax.contourf(X, Y, FD_solution, levels=40, cmap="coolwarm")
    ax.contour(contour_plot, linewidths=1, colors="k")
    plt.colorbar(contour_plot)
    plt.savefig(f"figures/{filename}_solution.svg")

    # Plot error
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    contour_plot = ax.contourf(X, Y, error, levels=40, cmap="coolwarm")
    ax.contour(contour_plot, linewidths=1, colors="k")
    plt.colorbar(contour_plot)
    plt.savefig(f"figures/{filename}_error.svg")


def plot_results_FV(filename):

    with open(f"results/{filename}.pkl", "rb") as f:
        results_dict = pickle.load(f)

    x = results_dict["x"]
    t = results_dict["t"]
    FV_solution = results_dict["FV_solution"]
    error = results_dict["error"]

    # Plot FV_solution
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    contour_plot = ax.contourf(x, t, FV_solution, levels=40, cmap="coolwarm")
    ax.contour(contour_plot, linewidths=1, colors="k")
    plt.colorbar(contour_plot)
    plt.savefig(f"figures/{filename}_solution.svg")

    # Plot error
    if error is not None:
        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        contour_plot = ax.contourf(x, t, error, levels=20, cmap="coolwarm")
        ax.contour(contour_plot, linewidths=1, colors="k")
        plt.colorbar(contour_plot)
        plt.savefig(f"figures/{filename}_error.svg")