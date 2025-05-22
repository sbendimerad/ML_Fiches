# 🗂️ PARTIE 1 — Expliquer un projet ML en entretien (RH et Tech)

## 🎯 Structure pour RH (version simplifiée, orientée métier)

- **Nom du projet :**  
  _Ex : Prédiction de départ à la concurrence_

- **Période :**  
  _Quand le projet a-t-il été réalisé ?_

- **Équipe :**  
  _Combien de personnes ont contribué ? Quel rôle ?_

- **Objectif du projet :**  
  _Ex : Réduire le churn client en prédisant les départs_

- **Contexte :**  
  - Données collectées sur un territoire spécifique  
  - Variables disponibles : historique client, usage, satisfaction, etc.  
  - Nombre de lignes / colonnes  
  - Avant : décisions prises a posteriori

- **Outils utilisés :**  
  - Python (Pandas, Scikit-learn)  
  - MLflow pour le suivi  
  - Résultats : fichiers / BDD / API

- **Utilisation des résultats :**  
  _Ex : Score de churn transmis aux équipes marketing_

- **ROI / Impact attendu :**  
  _Ex : Réduction de 15% du churn = économie de X€_

- **État actuel du projet :**  
  _En production / En test / En pause_

---

## 🧠 Structure pour Tech (plus détaillée)

- **Problèmes rencontrés :**
  - Valeurs manquantes : `IterativeImputer`, `KNNImputer`
  - Outliers : IQR / règles métier
  - Données déséquilibrées : SMOTE
  - Corrélation forte : PCA

- **Méthodologie ML :**
  - Baseline → modèles linéaires → modèles non-linéaires → modèles avancés

- **Algorithmes testés :**
  - DummyClassifier
  - Régression logistique
  - SVM Classifier
  - RandomForestClassifier
  - XGBoost

- **Validation :**
  - Cross-validation + GridSearchCV

- **Hyperparamètres :**
  - `n_estimators`, `max_depth`, `learning_rate`…

- **Métriques :**
  - Accuracy, Precision, Recall, F1-Score, Temps de calcul

---

# 🗂️ PARTIE 2 — Questions et Concepts ML

---

## 📌 1. Outliers

- **Détection simple :**
  - Boxplot / histogramme
  - Méthode IQR ou Z-score

- **Décision métier :**
  - Outlier ou pas selon le contexte

- **Traitement :**
  - Suppression
  - Winsorization
  - Log-transform
  - Modèles de détection : Isolation Forest / LOF

---

## 📌 2. Valeurs manquantes

- **Étape 1 — Analyse**
  - % de NaN
  - Visualisation `missingno`

- **Étape 2 — Imputation simple**
  - Numérique : moyenne / médiane
  - Catégorique : mode / "manquant"

- **Étape 3 — Imputation avancée**
  - `IterativeImputer`
  - `KNNImputer`
  - Nouvelle variable “is_missing”

---

## 📌 3. Démarche ML classique

1. Compréhension du problème
2. EDA
3. Nettoyage
4. Split train/test (stratify pour classification)
5. Baseline
6. Modèles progressifs (linéaires → arbres → boosting)
7. Hyperparamètres (GridSearchCV)
8. Évaluation (F1, MAE…)
9. Interprétation (SHAP, LIME)
10. Déploiement (pickle, FastAPI)

---

## 📌 4. Concepts fondamentaux

### 🔸 Overfitting / Underfitting
- **Overfitting :** bien sur train, mauvais sur test
- **Underfitting :** mauvais partout

### 🔸 Biais / Variance
- **Biais élevé :** modèle trop simple
- **Variance élevée :** modèle trop complexe

---

## 📌 5. Données déséquilibrées

- Ex : 95% classe 0
- **Problème :** Accuracy trop flatteuse
- **Solutions :**
  - Métriques : F1-score, recall
  - Techniques : SMOTE / undersampling

---

## 📌 6. Bagging vs Boosting

| Bagging                    | Boosting                      |
|---------------------------|-------------------------------|
| Arbres train parallèle    | Modèles entraînés en série   |
| Réduit variance           | Réduit biais                  |
| Exemple : Random Forest   | Exemple : XGBoost             |

---

## 📌 7. Random Forest (très simplifié)

- Entraîne plein d’arbres sur des échantillons différents
- Chaque arbre vote
- Fonctionne bien sans réglage complexe
- Se base sur **indice de Gini** pour les splits

---

## 📌 8. OneHotEncoder

- **Avantage :** ne suppose pas d'ordre
- **Inconvénient :** explosion de dimensions
- ➕ Bien pour arbres
- ➖ À combiner avec PCA si trop de colonnes

---

## 📌 9. Cross-validation

- Réduit les biais liés à un seul split
- Classique : `KFold` (ou `StratifiedKFold`)
- TimeSeries : `TimeSeriesSplit`

> _🖼️ Place ici une image du schéma de cross-validation_

---

## 📌 10. Feature Selection

- **Méthodes :**
  - Corrélation
  - Lasso (supprime)
  - Ridge (réduit)
  - SHAP / importance des features
- Réentraîner avec top K features

---

## 📌 11. Régression logistique

- Modèle linéaire + fonction **sigmoïde**
- Prédit une **probabilité**
- Seuil à 0.5 → classe 0 ou 1
- Interprétable (coefficients)

---

## 📌 12. Réseau de neurones (step-by-step)

1. Données passent dans la couche d’entrée
2. Multiplication des poids + ajout biais
3. Activation (ReLU, Sigmoid…)
4. Calcul d’une **perte** (loss)
5. **Backpropagation** : on propage l’erreur en arrière
6. **Mise à jour des poids** (descente de gradient)

### 📍 Vocabulaire
- **Epoch :** 1 passage complet sur le jeu d'entraînement
- **Batch :** portion des données
- **Forward pass :** prédiction
- **Backward pass :** correction

---

