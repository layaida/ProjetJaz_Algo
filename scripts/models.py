from pyspark.ml.clustering import KMeans, GaussianMixture, BisectingKMeans
from pyspark.sql import SparkSession

def train_kmeans(df, features_col="pca_features", k=3):
    kmeans = KMeans(k=k, seed=42, featuresCol=features_col)
    model = kmeans.fit(df)
    return model

def train_gaussian_mixture(df, features_col="pca_features", k=3):
    gm = GaussianMixture(k=k, seed=42, featuresCol=features_col)
    model = gm.fit(df)
    return model

def train_bisecting_kmeans(df, features_col="pca_features", k=3):
    bkm = BisectingKMeans(k=k, seed=42, featuresCol=features_col)
    model = bkm.fit(df)
    return model

