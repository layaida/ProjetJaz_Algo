from pyspark.sql import SparkSession 

def create_spark_session(app_name="CustomerSegmentation"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.local.dir", "C:/Users/djoud/CustomerSegmentation/tmp") \
        .config("spark.eventLog.dir", "file:///C:/Users/djoud/CustomerSegmentation/spark-events") \
        .getOrCreate()
    return spark
