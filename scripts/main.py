from pyspark_session import create_spark_session
from data_preprocessing import load_and_clean_data, enrich_data, prepare_features
from models import train_kmeans, train_gaussian_mixture, train_bisecting_kmeans
from evaluation import evaluate_model
from visualization import plot_clusters

def main():
    file_path = "file:///C:/Users/djoud/OneDrive/Bureau/M2 Big Data/JAZIRI/dataSet/database_p4.csv"
    
    # Étape 1 : Chargement et nettoyage des données
    print("Chargement et nettoyage des données...")
    df = load_and_clean_data(file_path)
    
    # Étape 2 : Enrichissement des données
    print("Enrichissement des données...")
    df = enrich_data(df)
    
    # Étape 3 : Préparation des features
    feature_cols = ["price", "freight_value", "product_weight_g", "product_length_cm"]
    print("Préparation des features...")
    df = prepare_features(df, feature_cols)
    
    # Étape 4 : Entraînement des modèles
    print("Entraînement des modèles...")
    kmeans_model = train_kmeans(df, k=4)
    gm_model = train_gaussian_mixture(df, k=4)
    bkm_model = train_bisecting_kmeans(df, k=4)

    
    # Étape 5 : Évaluation des modèles
    print("Évaluation des modèles...")
    kmeans_score = evaluate_model(kmeans_model, df)
    gm_score = evaluate_model(gm_model, df)
    bkm_score = evaluate_model(bkm_model, df)
    
    print(f"KMeans Silhouette Score: {kmeans_score}")
    print(f"Gaussian Mixture Silhouette Score: {gm_score}")
    print(f"Bisecting KMeans Silhouette Score: {bkm_score}")
    
    # Étape 6 : Visualisation des clusters pour KMeans
    print("Visualisation des clusters pour KMeans...")
    predictions = kmeans_model.transform(df)
    plot_clusters(predictions)

if __name__ == "__main__":
    main()
