from pyspark.ml.evaluation import ClusteringEvaluator

def evaluate_model(model, df, prediction_col="prediction"):
    predictions = model.transform(df)
    evaluator = ClusteringEvaluator(predictionCol=prediction_col)
    silhouette_score = evaluator.evaluate(predictions)
    return silhouette_score
