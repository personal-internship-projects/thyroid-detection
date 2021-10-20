import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
from src.model_operations import saveModel

from src.logger.auto_logger import autolog

class Kmeansclustering:

    def __init__(self):
        self.modelsDirs    = "src/models/kmeans-clustering.pkl"

    def elbowplot(self,data):
        wcss = []
        n = 25
        for i in range(1,n):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1,n),wcss,'bx-') # creating the graph between WCSS and the number of clusters
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig('src/dataset/preprocessed/K-Means_Elbow.PNG')
        self.kn = KneeLocator(range(1,n), wcss, curve='convex', direction='decreasing')
        print(wcss)
        return self.kn.knee



    def create_clusters(self,data,number_of_clusters):
        self.data = data
        autolog("Clustering started")
        self.kmeans      = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
        self.kmeans.fit(data)
        self.y_kmeans    = self.kmeans.predict(data)
        path             = self.modelsDirs 
        
        saveModel(path, self.kmeans)
        
        self.data['Cluster'] = self.y_kmeans
        

        autolog("Clustering Complered")
        return self.data

