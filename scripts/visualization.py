import matplotlib.pyplot as plt

def plot_clusters(spark_df, cluster_col="prediction", x_col="price", y_col="freight_value"):
    pdf = spark_df.select(x_col, y_col, cluster_col).toPandas()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(pdf[x_col], pdf[y_col], c=pdf[cluster_col], cmap='viridis')
    plt.title("Cluster Visualization")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()
