import time

def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"solving this differential equation took {end-start} s.")
    return wrapper