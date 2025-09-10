import cv2
import numpy as np
import random

# Fitness function: Otsu's variance
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

# GA main function
def genetic_algorithm(img, pop_size=30, generations=50, pc=0.8, pm=0.1):
    # Initialize population (random thresholds between 0–255)
    population = np.random.randint(0, 256, size=pop_size)
    
    for gen in range(generations):
        # Evaluate fitness
        fitness = [otsu_fitness(img, t) for t in population]

        # Selection (roulette wheel)
        total_fit = sum(fitness)
        probs = [f/total_fit for f in fitness]
        parents = np.random.choice(population, size=pop_size, p=probs)

        # Crossover
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[(i+1) % pop_size]
            if random.random() < pc:
                point = random.randint(1, 7)   # crossover at bit level
                mask = (1 << point) - 1
                child1 = (p1 & mask) | (p2 & ~mask)
                child2 = (p2 & mask) | (p1 & ~mask)
            else:
                child1, child2 = p1, p2
            offspring.extend([child1, child2])

        # Mutation
        for i in range(len(offspring)):
            if random.random() < pm:
                offspring[i] = random.randint(0, 255)

        # Replacement
        population = np.array(offspring)

    # Best solution
    best = max(population, key=lambda t: otsu_fitness(img, t))
    return best

# Example
if __name__ == "__main__":
    img = cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE)
    best_t = genetic_algorithm(img)
    print("Best threshold (GA):", best_t)
    _, segmented = cv2.threshold(img, best_t, 255, cv2.THRESH_BINARY)
    cv2.imshow("Original", img)
    cv2.imshow("GA Thresholded", segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
