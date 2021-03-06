import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import pandas as pd
from src.model_operations import saveModel
from sklearn.metrics import silhouette_score
from src.logger.auto_logger import autolog
import logging


class Kmeansclustering:

    def __init__(self):
       
        self.modelsDirs = "src/models/kmeans-clustering.pkl"
        logging.getLogger('matplotlib').setLevel(logging.ERROR)

    def elbowplot(self, data):
        """
        This function plots the elbow plot to find the optimal number of clusters

        Args:
            data ([pandas.DataFrame]): [Stores all the feature columns of the data]

        Returns:
            [int]: [Returns the optimal number of clusters or knee value]
        """
        self.wcss = []
        self.n = range(2, 15)
        for i in self.n:
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(data)
            self.wcss.append(kmeans.inertia_)
        # creating the graph between WCSS and the number of clusters
        plt.plot(self.n, self.wcss, marker="+")
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig('src/dataset/preprocessed/K-Means_Elbow.PNG')
        d = pd.DataFrame({'No.of_Clusters': self.n, 'WSS': self.wcss})
        # plt.scatter(x = "No.of_Clusters", y = 'WSS', data = d, marker="+")
        # plt.savefig('src/dataset/preprocessed/WSS_Elbow.PNG')
        self.kn = KneeLocator(self.n, self.wcss, curve='convex',
                              direction='decreasing', polynomial_degree=8)
        print(self.wcss)
        print(self.kn.knee)
        return self.kn.knee

    def silhoutee_scores(self, data):
        """
        This function calculates the silhouette score for the given data

        Args:
            data ([pandas.DataFrame]): [Stores all the feature columns of the data]
        """
        self.s_score = []
        #self.data = data
        #data.to_csv("ff.csv",index= None, header=True)
        #autolog("Clustering started")
        for i in self.n:
            #self.kmeans      = KMeans(n_clusters=i, init='k-means++', random_state=42)
            # self.kmeans.fit(data)
            labels = KMeans(n_clusters=i, init="k-means++",
                            random_state=200).fit(data).labels_
            print("Silhouette score for k(clusters) = "+str(i)+" is "
                  + str(silhouette_score(data, labels, sample_size=5000, random_state=200)))
            self.s_score.append(silhouette_score(
                data, labels, sample_size=5000, random_state=200))

    def scores_clustering(self):
        """
        This function calculates the silhouette score for the given data and stores everythin into a CSV file

        """
        mycenters = pd.DataFrame(
            {'No.of_Clusters': self.n, 'WSS': self.wcss, 'Silhouette_score': self.s_score})
        mycenters.to_csv("scores.csv", index=None, header=True)

    def create_clusters(self, data, number_of_clusters):
        """
        This function creates the clusters based on the elbow plot

        Args:
            data ([pandas.DataFrame]): [Stores all the feature columns of the data]  
            number_of_clusters ([int]): [parameter value for k-means algorithm for creating clusters]

        Returns:
            [pandas.DataFrame]: [Stores the cluster labels for each row of the data]
        """
        self.data = data
        self.kmeans = KMeans(n_clusters=number_of_clusters,
                             init='k-means++', random_state=42)
        self.kmeans.fit(data)
        self.y_kmeans = self.kmeans.predict(data)
        self.data['Cluster'] = self.y_kmeans

        path = f"{self.modelsDirs}"
        saveModel(path, self.kmeans)

        return self.data
