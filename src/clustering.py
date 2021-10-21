import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
from src.model_operations import saveModel
from sklearn.metrics import silhouette_samples, silhouette_score
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
        self.kn = KneeLocator(range(1,n), wcss, curve='convex', direction='decreasing', polynomial_degree=8)
        print(wcss)
        return self.kn.knee



    def create_clusters(self,data):
        #self.data = data
        #data.to_csv("ff.csv",index= None, header=True)
        #autolog("Clustering started")
        for i in range(3,13):
            #self.kmeans      = KMeans(n_clusters=i, init='k-means++', random_state=42)
            #self.kmeans.fit(data)
            labels=KMeans(n_clusters=i,init="k-means++",random_state=200).fit(data).labels_
            print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(silhouette_score(data,labels,metric="euclidean",sample_size=1000,random_state=200)))
