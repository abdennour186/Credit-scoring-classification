# Projet de Classification de Crédit Scoring

## Description

Ce projet consiste en la construction et l'évaluation de modèles de classification pour prédire le statut de crédit à partir de jeux de données. Nous avons utilisé différentes techniques de prétraitement des données, d'apprentissage supervisé et d'évaluation des modèles pour optimiser les performances de classification.

## Objectifs

1. **Apprentissage supervisé : Feature Engineering et Classification**
   - **Chargement et Préparation des Données** : Importation des données, transformation des données en tableaux numpy, séparation des variables caractéristiques et de la variable à prédire.
   - **Apprentissage et Évaluation des Modèles** : Application des algorithmes de classification (Arbre CART, K-plus-proches-voisins, MultilayerPerceptron) et évaluation à l'aide de la matrice de confusion, de l'accuracy, et des critères de rappel et précision.
   - **Normalisation des Données** : Utilisation de `StandardScaler` ou `MinMaxScaler` pour normaliser les données, et comparaison des résultats avant et après normalisation.
   - **Création de Nouvelles Variables Caractéristiques** : Application de l'Analyse en Composantes Principales (ACP) pour réduire la dimensionnalité et amélioration des performances des modèles.
   - **Sélection de Variables** : Utilisation de `RandomForestClassifier` pour déterminer l'importance des variables et sélection des meilleures variables pour la prédiction.
   - **Paramétrage des Classifieurs** : Utilisation de `GridSearchCV` pour optimiser les hyperparamètres du meilleur modèle.
   - **Création d'un Pipeline** : Automatisation du processus de prétraitement et de classification avec un pipeline `scikit-learn`, et sauvegarde du pipeline pour une utilisation future.
   - **Comparaison de Plusieurs Algorithmes** : Évaluation et comparaison de divers algorithmes de classification (NaiveBayesSimple, Arbre CART, ID3, Decision Stump, etc.) en termes d'accuracy, AUC, et temps d'exécution.

2. **Apprentissage Supervisé : Données Hétérogènes**
   - **Préparation des Données** : Importation et traitement des données hétérogènes (numériques et catégorielles), traitement des valeurs manquantes, et transformation des variables catégorielles en variables binaires avec `OneHotEncoder`.
   - **Comparaison des Algorithmes** : Exécution de la fonction de comparaison sur les données traitées pour évaluer les performances des algorithmes sur les données hétérogènes.

## Technologies et Bibliothèques Utilisées

- **Pandas** : Pour le chargement et la manipulation des données.
- **NumPy** : Pour les transformations et le prétraitement des données.
- **Scikit-learn** : Pour les algorithmes d'apprentissage supervisé, la normalisation des données, la création du pipeline, et la sélection des variables.
- **Matplotlib** : Pour la visualisation des résultats (histogrammes des importances des variables, courbes d'évolution de l'accuracy).
- **GridSearchCV** : Pour l'optimisation des hyperparamètres des modèles.
- **OneHotEncoder** : Pour la transformation des variables catégorielles.
