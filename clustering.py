import sklearn


def KMeans(Vectors, Clusters):
    Model = sklearn.cluster.KMeans(n_clusters = Clusters)
    Results = Model.fit_predict(Vectors)
    S = sklearn.metrics.silhouette_score(Vectors, Results)
    CH = sklearn.metrics.calinski_harabasz_score(Vectors, Results)
    DB = sklearn.metrics.davies_bouldin_score(Vectors, Results)
    return Model, Results, S, CH, DB


def HDBSCAN(Vectors, MinClusterSize):
    Model = sklearn.cluster.HDBSCAN(min_cluster_size = MinClusterSize)
    Results = Model.fit_predict(Vectors)
    S = sklearn.metrics.silhouette_score(Vectors, Results)
    CH = sklearn.metrics.calinski_harabasz_score(Vectors, Results)
    DB = sklearn.metrics.davies_bouldin_score(Vectors, Results)
    return Model, Results, S, CH, DB    

    
def AgglomerativeClustering(Vectors, Clusters):
    Model = sklearn.cluster.AgglomerativeClustering(n_clusters = Clusters, metric = 'cosine', linkage = 'average')
    Results = Model.fit_predict(Vectors)
    S = sklearn.metrics.silhouette_score(Vectors, Results)
    CH = sklearn.metrics.calinski_harabasz_score(Vectors, Results)
    DB = sklearn.metrics.davies_bouldin_score(Vectors, Results)
    return Model, Results, S, CH, DB


def SpectralClustering(Vectors, Clusters):
    Model = sklearn.cluster.SpectralClustering(n_clusters = Clusters)
    Results = Model.fit_predict(Vectors)
    S = sklearn.metrics.silhouette_score(Vectors, Results)
    CH = sklearn.metrics.calinski_harabasz_score(Vectors, Results)
    DB = sklearn.metrics.davies_bouldin_score(Vectors, Results)
    return Model, Results, S, CH, DB
