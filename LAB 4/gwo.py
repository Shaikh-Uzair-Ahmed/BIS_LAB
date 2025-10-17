import numpy as np

def sphere(x):
    return np.sum(x**2)

class GreyWolfOptimizer:
    def __init__(self, obj_func, n_wolves, dim, max_iter, lb=-10, ub=10):
        self.obj_func = obj_func
        self.n_wolves = n_wolves
        self.dim = dim
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub

        self.positions = np.random.uniform(self.lb, self.ub, (self.n_wolves, self.dim))

        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')

        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('inf')

        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('inf')

    def optimize(self):
        for iter in range(self.max_iter):
            for i in range(self.n_wolves):
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

                fitness = self.obj_func(self.positions[i])

                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()

            a = 2 - iter * (2 / self.max_iter)

            for i in range(self.n_wolves):
                for j in range(self.dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    self.positions[i, j] = (X1 + X2 + X3) / 3

        return self.alpha_pos, self.alpha_score


if __name__ == "__main__":
    # Take inputs from user
    n_wolves = int(input("Enter number of wolves: "))
    dim = int(input("Enter number of dimensions: "))
    max_iter = int(input("Enter max iterations: "))

    gwo = GreyWolfOptimizer(obj_func=sphere, n_wolves=n_wolves, dim=dim, max_iter=max_iter)
    best_pos, best_score = gwo.optimize()

    print(f"Best Position: {best_pos}")
    print(f"Best Score: {best_score}")
