from pyspark_session import create_spark_session
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
import random

def load_and_clean_data(file_path):
    spark = create_spark_session()
    df = spark.read.csv(file_path, header=True, inferSchema=True, sep="\t")
    
    if 'order_status' not in df.columns:
        raise ValueError("La colonne 'order_status' est absente du fichier CSV.")
    
    df = df.filter(df.order_status == 'delivered') \
           .na.fill({"price": 0, "freight_value": 0, "product_weight_g": 0, "product_length_cm": 0})
    return df

def enrich_data(df):
    df = df.withColumn("customer_age_group", F.lit(random.choice(["18-25", "26-35", "36-50", "50+"])))
    df = df.withColumn("purchase_history", F.lit(random.randint(1, 20)))
    return df

def prepare_features(df, feature_cols, output_col="features"):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol=output_col)
    df = assembler.transform(df)
    
    scaler = StandardScaler(inputCol=output_col, outputCol="scaled_features")
    df = scaler.fit(df).transform(df)
    
    pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")
    df = pca.fit(df).transform(df)
    return df
