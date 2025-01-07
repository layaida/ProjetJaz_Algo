from pyspark_session import create_spark_session
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA

# Fonction pour charger et nettoyer les données
def load_and_clean_data(file_path):
    spark = create_spark_session()
    
    # Charger le fichier CSV avec un séparateur explicite
    df = spark.read.csv(file_path, header=True, inferSchema=True, sep="\t")
    
    # Vérifier si la colonne 'order_status' existe
    if 'order_status' not in df.columns:
        raise ValueError("La colonne 'order_status' est absente du fichier CSV.")
    
    # Filtrer les données pour ne garder que les commandes livrées
    df = df.filter(df.order_status == 'delivered')

    # Remplissage des valeurs nulles
    df = df.na.fill({"price": 0, "freight_value": 0, "product_weight_g": 0, "product_length_cm": 0})

    print("Colonnes chargées :", df.columns)
    return df

# Fonction pour préparer les features
def prepare_features(df, feature_cols, output_col="features"):
    # Vérifier et traiter les valeurs nulles dans les colonnes utilisées pour les features
    print("Nombre de valeurs nulles par colonne :")
    df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in feature_cols]).show()

    # Supprimer les lignes avec des valeurs nulles
    df = df.dropna(subset=feature_cols)

    # Utiliser VectorAssembler pour assembler les features
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol=output_col,
        handleInvalid="skip"
    )
    df = assembler.transform(df)

    # Standardisation des données
    scaler = StandardScaler(inputCol=output_col, outputCol="scaled_features", withStd=True, withMean=True)
    scaled_data = scaler.fit(df).transform(df)

    # Réduction de la dimension avec PCA
    pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")
    pca_data = pca.fit(scaled_data).transform(scaled_data)

    return pca_data