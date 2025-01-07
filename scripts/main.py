from pyspark_session import create_spark_session 
from data_preprocessing import load_and_clean_data, prepare_features
from models import train_kmeans, train_gaussian_mixture
from evaluation import evaluate_model
from visualization import plot_clusters

def main():
    # Chemin du fichier (mettez à jour selon votre système)
    file_path = "file:///C:/Users/djoud/OneDrive/Bureau/M2 Big Data/JAZIRI/dataSet/database_p4.csv"
    
    try:
        # Étape 1 : Charger et nettoyer les données
        print("Chargement et nettoyage des données...")
        df = load_and_clean_data(file_path)
        
        # Définir les colonnes des features
        feature_cols = ["price", "freight_value", "product_weight_g", "product_length_cm"]
        
        print("Préparation des features...")
        df = prepare_features(df, feature_cols)
        
        # Vérification rapide des données transformées
        print("Échantillon des données transformées :")
        df.select("features").show(truncate=False)
        
        # Étape 2 : Entraînement des modèles
        print("Entraînement du modèle KMeans...")
        kmeans_model = train_kmeans(df, k=4)
        
        print("Entraînement du modèle Gaussian Mixture...")
        gm_model = train_gaussian_mixture(df, k=4)
        
        # Étape 3 : Évaluation des modèles
        print("Évaluation des modèles...")
        kmeans_score = evaluate_model(kmeans_model, df)
        gm_score = evaluate_model(gm_model, df)
        
        # Affichage des résultats
        print(f"KMeans Silhouette Score: {kmeans_score}")
        print(f"Gaussian Mixture Silhouette Score: {gm_score}")
        
        # Choisir le meilleur modèle
        best_model = "KMeans" if kmeans_score > gm_score else "Gaussian Mixture"
        print(f"Le meilleur modèle est : {best_model}")
        
        # Étape 4 : Visualisation des clusters
        print("Visualisation des clusters pour le modèle choisi...")
        if best_model == "KMeans":
            predictions = kmeans_model.transform(df)
        else:
            predictions = gm_model.transform(df)
        
        plot_clusters(predictions)
    
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    main()
