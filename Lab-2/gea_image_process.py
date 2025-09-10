import cv2
import numpy as np
import random

# Fitness function (same as GA)
def otsu_fitness(img, threshold):
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    total_pixels = img.size
    prob = hist / total_pixels
    w0 = np.sum(prob[:threshold+1])
    w1 = np.sum(prob[threshold+1:])
    if w0 == 0 or w1 == 0: return 0
    mu0 = np.sum(np.arange(0, threshold+1) * prob[:threshold+1]) / w0
    mu1 = np.sum(np.arange(threshold+1, 256) * prob[threshold+1:]) / w1
    return w0 * w1 * (mu0 - mu1) ** 2

# Expression function: translate gene sequence into threshold
def express(sequence):
    return sum(sequence) % 256   # map sequence into [0,255]

# GEA main function
def gene_expression_algorithm(img, pop_size=30, generations=50, pc=0.8, pm=0.1, gene_len=8):
    # Initialize population: each is a sequence of integers (genes)
    population = [np.random.randint(0, 100, size=gene_len).tolist() for _ in range(pop_size)]

    # Track best
    best_seq = None
    best_fit = -1

    for gen in range(generations):
        # Evaluate fitness
        fitness = []
        for seq in population:
            threshold = express(seq)
            score = otsu_fitness(img, threshold)
            fitness.append(score)
            if score > best_fit:
                best_fit = score
                best_seq = seq

        # Selection (tournament)
        parents = []
        for _ in range(pop_size):
            i, j = random.sample(range(pop_size), 2)
            parents.append(population[i] if fitness[i] > fitness[j] else population[j])

        # Crossover
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i][:], parents[(i+1) % pop_size][:]
            if random.random() < pc:
                point = random.randint(1, gene_len-1)
                child1 = p1[:point] + p2[point:]
                child2 = p2[:point] + p1[point:]
            else:
                child1, child2 = p1, p2
            offspring.extend([child1, child2])

        # Mutation
        for child in offspring:
            for g in range(gene_len):
                if random.random() < pm:
                    child[g] = random.randint(0, 100)

        # Replacement
        population = offspring

    # Final best threshold
    best_threshold = express(best_seq)
    return best_threshold

# Example
if __name__ == "__main__":
    img = cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE)
    best_t = gene_expression_algorithm(img)
    print("Best threshold (GEA):", best_t)
    _, segmented = cv2.threshold(img, best_t, 255, cv2.THRESH_BINARY)
    cv2.imshow("Original", img)
    cv2.imshow("GEA Thresholded", segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
