import cv2
import numpy as np
import random

# -----------------------------
# Otsu’s fitness function
# -----------------------------
def otsu_fitness(img, threshold):
    # Apply threshold
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    total_pixels = img.size

    # Probabilities
    prob = hist / total_pixels

    # Class probabilities and means
    w0 = np.sum(prob[:threshold+1])  # background weight
    w1 = np.sum(prob[threshold+1:])  # foreground weight
    if w0 == 0 or w1 == 0:
        return 0  # invalid threshold

    mu0 = np.sum(np.arange(0, threshold+1) * prob[:threshold+1]) / w0
    mu1 = np.sum(np.arange(threshold+1, 256) * prob[threshold+1:]) / w1

    # Between-class variance
    variance = w0 * w1 * (mu0 - mu1) ** 2
    return variance


# -----------------------------
# PSO Algorithm
# -----------------------------
def pso_thresholding(img, num_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5):
    # Initialize particles
    positions = np.random.randint(0, 256, size=num_particles)  # thresholds
    velocities = np.random.uniform(-1, 1, size=num_particles)
    pbest_positions = positions.copy()
    pbest_scores = np.array([otsu_fitness(img, t) for t in positions])

    # Global best
    gbest_index = np.argmax(pbest_scores)
    gbest_position = pbest_positions[gbest_index]
    gbest_score = pbest_scores[gbest_index]

    # Iterations
    for _ in range(max_iter):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = random.random(), random.random()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )

            # Update position (threshold must stay in [0,255])
            positions[i] = int(np.clip(positions[i] + velocities[i], 0, 255))

            # Evaluate fitness
            score = otsu_fitness(img, positions[i])

            # Update personal best
            if score > pbest_scores[i]:
                pbest_positions[i] = positions[i]
                pbest_scores[i] = score

        # Update global best
        gbest_index = np.argmax(pbest_scores)
        if pbest_scores[gbest_index] > gbest_score:
            gbest_score = pbest_scores[gbest_index]
            gbest_position = pbest_positions[gbest_index]

    return gbest_position


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Load grayscale image
    img = cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE)

    # Run PSO to find best threshold
    best_threshold = pso_thresholding(img)
    print("Best Threshold Found:", best_threshold)

    # Apply threshold
    _, segmented = cv2.threshold(img, best_threshold, 255, cv2.THRESH_BINARY)

    # Show results
    cv2.imshow("Original", img)
    cv2.imshow("Segmented (PSO)", segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
