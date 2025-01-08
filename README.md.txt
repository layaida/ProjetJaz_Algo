# **README - Projet de Segmentation des Clients avec PySpark**

## **Présentation générale du projet**

Ce projet a pour objectif de réaliser une segmentation des clients à l'aide de techniques de clustering, en exploitant la puissance de PySpark pour traiter des volumes de données importants. En analysant les comportements d'achat et les caractéristiques des commandes, le projet vise à identifier des groupes homogènes de clients afin d'aider les entreprises à mieux cibler leurs stratégies marketing, logistiques et commerciales.

## **Contexte**

Dans le secteur du commerce électronique, comprendre le comportement des clients est un facteur clé de succès. Cependant, les volumes de données générés par les transactions quotidiennes peuvent rapidement devenir ingérables avec des outils conventionnels. PySpark, une librairie Big Data puissante, permet de traiter et d'analyser efficacement ces données massives. Ce projet exploite ces capacités pour effectuer une segmentation client basée sur des caractéristiques telles que les prix des produits, les frais de livraison ou encore les caractéristiques physiques des articles.

## **Problématique**

Comment segmenter efficacement une base de clients en exploitant des données transactionnelles, afin de découvrir des schémas cachés et d'optimiser les stratégies commerciales et logistiques ?

Les défis principaux incluent :
- Le nettoyage et la préparation des données pour éliminer les valeurs manquantes et les incohérences.
- La réduction de la dimensionnalité pour simplifier l'analyse tout en conservant l'essentiel de l'information.
- L'entraînement et la comparaison de plusieurs modèles de clustering, tels que K-Means, Gaussian Mixture, DBSCAN, et Bisecting K-Means.
- La visualisation des résultats afin de mieux comprendre les différents clusters.

## **Déroulement du projet**

### **1. Fichier `pyspark_session.py`**
Ce fichier contient la fonction **`create_spark_session`** qui initialise une session PySpark avec les configurations nécessaires :
- Allocation de mémoire pour le driver et les exécuteurs.
- Définition des répertoires locaux pour le stockage temporaire et les journaux d'événements.

### **2. Fichier `data_preprocessing.py`**
Ce module s'occupe du prétraitement des données :
- **Chargement des données :** Les données sont importées depuis un fichier CSV avec un séparateur tabulation.
- **Nettoyage :** Les valeurs manquantes dans les colonnes critiques sont remplacées par des valeurs par défaut, et les commandes non livrées sont filtrées.
- **Enrichissement :** Une étape supplémentaire ajoute des informations calculées à partir des colonnes existantes.
- **Assemblage des features :** Les colonnes pertinentes (égales à `price`, `freight_value`, etc.) sont combinées en vecteurs exploitables pour les modèles de clustering.
- **Réduction de dimension :** Une PCA (Analyse en Composantes Principales) réduit les dimensions à 3 composantes principales pour optimiser les performances des modèles.

### **3. Fichier `models.py`**
Ce module implémente différents modèles de clustering :
- **K-Means :** Divise les données en clusters basés sur des centres calculés itérativement.
- **Gaussian Mixture :** Adapte une approche probabiliste pour modéliser les clusters comme des distributions normales.
- **DBSCAN :** Identifie les clusters à densité élevée, ce qui est utile pour détecter les anomalies ou les cas isolés.
- **Bisecting K-Means :** Une variante hiérarchique de K-Means, mieux adaptée aux clusters imbriqués.

Chaque modèle est entraîné sur les mêmes features pour assurer une comparaison équitable.

### **4. Fichier `evaluation.py`**
Ce fichier gère l'évaluation des modèles :
- **Silhouette Score :** Évalue la cohésion interne des clusters.
- **Davies-Bouldin Index :** Mesure la compacité et la séparation des clusters.
- **Calinski-Harabasz Index :** Quantifie le rapport entre la dispersion intra-cluster et inter-cluster.

### **5. Fichier `visualization.py`**
Permet de visualiser les clusters en utilisant Matplotlib :
- Les clusters sont représentés sous forme de nuages de points sur les dimensions les plus pertinentes (par exemple, `price` et `freight_value`).
- Cette visualisation facilite l'interprétation des différents segments clients.

### **6. Fichier `main.py`**
Le fichier principal orchestre les étapes du projet :
1. Chargement et prétraitement des données.
2. Entraînement des modèles et évaluation comparative.
3. Visualisation des résultats.
4. Affichage des scores pour identifier le modèle le plus performant.

## **Résultats**

L'analyse a permis d'identifier des clusters bien définis, comme illustré dans la visualisation ci-dessus. Voici quelques conclusions importantes :
- Le modèle K-Means a obtenu un bon équilibre entre rapidité d'exécution et qualité des clusters, avec un Silhouette Score élevé.
- Gaussian Mixture a été utile pour capturer les clusters avec des formes plus complexes.
- DBSCAN a mis en évidence des anomalies et des points isolés, ce qui peut être intéressant pour l'optimisation logistique.
- Bisecting K-Means s'est montré performant pour des clusters imbriqués mais plus long à exécuter.

## **Conclusion**

Ce projet met en évidence l'importance des outils Big Data comme PySpark pour traiter et analyser efficacement de grandes bases de données. Les résultats obtenus montrent le potentiel du clustering pour segmenter les clients et découvrir des insights exploitables. Les entreprises peuvent utiliser ces clusters pour adapter leurs stratégies marketing et améliorer leurs performances logistiques.

### **Améliorations possibles**
- Ajouter d'autres métriques de qualité des clusters pour affiner l'évaluation.
- Enrichir les données avec des variables exogènes (par exemple, données démographiques).
- Automatiser l’optimisation des hyperparamètres des modèles.
- Implémenter des techniques de clustering hiérarchique avancées pour explorer différents niveaux de segmentation.

