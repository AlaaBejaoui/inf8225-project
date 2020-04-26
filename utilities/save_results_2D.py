import os
import sys
sys.path.append(os.getcwd())
import pickle

def save_results(filename, X, Y, nn_solution, exact_solution, error, training_loss):
    results_dict = {}

    results_dict["X"] = X
    results_dict["Y"] = Y
    results_dict["nn_solution"] = nn_solution
    results_dict["exact_solution"] = exact_solution
    results_dict["error"] = error
    results_dict["training_loss"] = training_loss

    with open(f"results/{filename}.pkl", "wb") as f:
        pickle.dump(results_dict, f)

    
    