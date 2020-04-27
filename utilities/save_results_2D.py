import os
import sys
sys.path.append(os.getcwd())
import pickle


def save_results_hyperbolic(filename, x, t, nn_solution, exact_solution, error, training_loss):
    results_dict = {}

    results_dict["x"] = x
    results_dict["t"] = t
    results_dict["nn_solution"] = nn_solution
    results_dict["exact_solution"] = exact_solution
    results_dict["error"] = error
    results_dict["training_loss"] = training_loss

    with open(f"results/{filename}.pkl", "wb") as f:
        pickle.dump(results_dict, f)


def save_results_elliptic(filename, X, Y, nn_solution, exact_solution, error, training_loss):
    results_dict = {}

    results_dict["X"] = X
    results_dict["Y"] = Y
    results_dict["nn_solution"] = nn_solution
    results_dict["exact_solution"] = exact_solution
    results_dict["error"] = error
    results_dict["training_loss"] = training_loss

    with open(f"results/{filename}.pkl", "wb") as f:
        pickle.dump(results_dict, f)


def save_results_FD(filename, X, Y, FD_solution, error):
    results_dict = {}

    results_dict["X"] = X
    results_dict["Y"] = Y
    results_dict["FD_solution"] = FD_solution
    results_dict["error"] = error

    with open(f"results/{filename}.pkl", "wb") as f:
        pickle.dump(results_dict, f)