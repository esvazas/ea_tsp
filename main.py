from solver.solver import TSP_solver
if __name__ == "__main__":
    ea = TSP_solver()
    ea.optimize("data/tour29.csv")