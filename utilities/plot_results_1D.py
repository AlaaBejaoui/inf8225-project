import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append(os.getcwd())


def plot_results(filename):

    with open(f"results/{filename}.pkl", "rb") as f:
        results_dict = pickle.load(f)

    x = results_dict["x"]
    nn_solution = results_dict["nn_solution"]
    exact_solution = results_dict["exact_solution"]
    error = results_dict["error"]
    training_loss = results_dict["training_loss"]

    # Plot NN solution and exact solution
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()
    ax.plot(x, nn_solution, "r", label="NN solution")
    ax.plot(x, exact_solution, "k", label="Exact solution")
    ax.legend()
    plt.savefig(f"figures/{filename}_solution.svg")

    # Plot error
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("Error")
    ax.grid()
    ax.plot(x, error, "k")
    plt.savefig(f"figures/{filename}_error.svg")

    # Plot loss
    fig, ax = plt.subplots()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.semilogy(training_loss)
    plt.savefig(f"figures/{filename}_loss.svg")
