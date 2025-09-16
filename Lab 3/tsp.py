import numpy as np

class AntColony:
    def __init__(self, distance_matrix, n_ants, n_iterations, decay, alpha=1, beta=2):
        self.dist_matrix = distance_matrix
        self.pheromone = np.ones(self.dist_matrix.shape) / len(distance_matrix)
        self.all_inds = range(len(distance_matrix))
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.construct_all_paths()
            self.spread_pheromone(all_paths, self.n_ants)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path            
            self.pheromone *= self.decay

            # Show results after each iteration
            print(f"Iteration {i+1}: shortest path {shortest_path[0]} with distance {shortest_path[1]:.4f}")

            # Print pheromone matrix if you want to check pheromone levels each iteration
            print("Pheromone levels:")
            print(self.get_pheromone_levels())
            print("-" * 40)

        return all_time_shortest_path

    def construct_all_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.construct_path(0)
            all_paths.append((path, self.path_distance(path)))
        return all_paths

    def construct_path(self, start):
        path = [start]
        visited = set(path)
        for _ in range(len(self.dist_matrix) - 1):
            move = self.pick_move(path[-1], visited)
            path.append(move)
            visited.add(move)
        return path

    def pick_move(self, current, visited):
        pheromone = self.pheromone[current]
        distances = self.dist_matrix[current]
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        heuristic = 1.0 / (distances + 1e-10)
        heuristic[list(visited)] = 0

        prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
        total = prob.sum()
        if total == 0:
            candidates = list(set(self.all_inds) - visited)
            return np.random.choice(candidates)
        prob = prob / total
        move = np.random.choice(self.all_inds, p=prob)
        return move

    def path_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.dist_matrix[path[i]][path[i + 1]]
        distance += self.dist_matrix[path[-1]][path[0]]
        return distance

    def spread_pheromone(self, all_paths, n_ants):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_ants]:
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += 1.0 / dist
            self.pheromone[path[-1]][path[0]] += 1.0 / dist

    def get_pheromone_levels(self):
        return self.pheromone.copy()


def get_distance_matrix():
    n = int(input("Enter the number of cities: "))
    print("Enter the distance matrix row by row (space separated):")
    matrix = []
    for i in range(n):
        while True:
            row = input(f"Row {i+1}: ").strip().split()
            if len(row) != n:
                print(f"Please enter exactly {n} values.")
                continue
            try:
                row = list(map(float, row))
                matrix.append(row)
                break
            except ValueError:
                print("Please enter valid numeric values.")
    return np.array(matrix)


if __name__ == "__main__":
    dist_matrix = get_distance_matrix()
    n_ants = int(input("Enter the number of ants: "))
    n_iterations = int(input("Enter the number of iterations: "))
    decay = float(input("Enter pheromone decay rate (0 < decay < 1): "))
    alpha = float(input("Enter alpha (pheromone influence): "))
    beta = float(input("Enter beta (heuristic influence): "))

    ant_colony = AntColony(dist_matrix, n_ants, n_iterations, decay, alpha, beta)
    best_path, best_dist = ant_colony.run()

    print("\nBest overall path:", best_path)
    print("Best overall distance:", best_dist)
