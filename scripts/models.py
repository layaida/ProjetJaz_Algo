from pyspark.ml.clustering import KMeans, GaussianMixture 

def train_kmeans(df, features_col="features", k=3):
    kmeans = KMeans(k=k, seed=42, featuresCol=features_col)
    model = kmeans.fit(df)
    return model

def train_gaussian_mixture(df, features_col="features", k=3):
    gm = GaussianMixture(k=k, seed=42, featuresCol=features_col)
    model = gm.fit(df)
    return model
