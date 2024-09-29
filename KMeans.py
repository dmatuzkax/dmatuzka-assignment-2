import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt

class KMeans():
    def __init__(self, data, k, method='random', centers = []):
        self.data = data
        self.k = k
        self.assignment = [-1 for _ in range(len(data))]
        self.snaps = []
        self.converged = False
        self.method = method 
        self.centers = np.array(centers)

    def snap(self, centers):
        TEMPFILE = "temp.png"
        plt.figure(figsize=(9, 6))  # Match the figure size

        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.assignment, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Centroids')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=12, ncol=1)

        plt.xlim(-11, 11)
        plt.ylim(-11, 11)
        plt.title('KMeans Clustering Animation')
        plt.xticks(np.arange(-10, 11, 5))  
        plt.yticks(np.arange(-10, 11, 5))  
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        plt.savefig(TEMPFILE)
        
        plt.close()

        self.snaps.append(im.fromarray(np.asarray(im.open(TEMPFILE))))

    def isunassigned(self, i):
        return self.assignment[i] == -1

    def initialize(self):
        if self.method == 'random':
            return self.data[np.random.choice(len(self.data), size=self.k, replace=False)]
        elif self.method == 'farthest':
            return self.farthest_first()
        elif self.method == 'kmeans++':
            return self.kmeans_plus_plus()
        elif self.method == 'manual':
            return self.centers
        
    def farthest_first(self):
        centers = [self.data[np.random.randint(len(self.data))]]
        
        for _ in range(1, self.k):
            dists = np.array([min(np.linalg.norm(p - c) for c in centers) for p in self.data])
            
            farthest_point = self.data[np.argmax(dists)]
            
            centers.append(farthest_point)
        
        return np.array(centers)

    def kmeans_plus_plus(self):
        centers = [self.data[np.random.randint(len(self.data))]]

        for _ in range(1, self.k):

            dists = np.array([min(np.linalg.norm(p - c) ** 2 for c in centers) for p in self.data])
            
            probabilities = dists / dists.sum()

            next_center = self.data[np.random.choice(len(self.data), p=probabilities)]
            centers.append(next_center)

        return np.array(centers)

    def make_clusters(self, centers):
        for i in range(len(self.assignment)):
            for j in range(self.k):
                if self.isunassigned(i):
                    self.assignment[i] = j
                    dist = self.dist(centers[j], self.data[i])
                else:
                    new_dist = self.dist(centers[j], self.data[i])
                    if new_dist < dist:
                        self.assignment[i] = j
                        dist = new_dist

    def compute_centers(self):
        centers = []
        for i in range(self.k):
            cluster = []
            for j in range(len(self.assignment)):
                if self.assignment[j] == i:
                    cluster.append(self.data[j])
            centers.append(np.mean(np.array(cluster), axis=0))

        return np.array(centers)
    
    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        for i in range(self.k):
            if self.dist(centers[i], new_centers[i]) != 0:
                return True
        return False

    def dist(self, x, y):
        return np.linalg.norm(x - y)

    def lloyds(self):
        centers = self.initialize()
        self.make_clusters(centers)
        self.snap(centers)
        new_centers = self.compute_centers()
        self.snap(new_centers)

        while self.are_diff(centers, new_centers):
            self.unassign()
            centers = new_centers
            self.make_clusters(centers)
            new_centers = self.compute_centers()
            self.snap(new_centers)
        return
