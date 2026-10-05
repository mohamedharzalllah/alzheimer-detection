# Détection d'Alzheimer — Projet Deep Learning (M2 BDIA 2025)

Projet réalisé dans le cadre de la compétition Kaggle **« M-2 BDIA DL Project 2025 »** : prédire si un individu est **sain (0)** ou **atteint d'Alzheimer (1)** à partir de caractéristiques d'écriture manuscrite.

Deux approches sont comparées : un pipeline de **Machine Learning classique** (stacking d'ensembles) et un pipeline de **Deep Learning** (MLP résiduel avec focal loss).

## Structure du projet

```
projet DL/
├── data/
│   ├── data.csv               # Jeu d'entraînement (104 patients × 450 features + cible `class`), séparateur ','
│   ├── test.csv               # Jeu de test (70 patients × 450 features), séparateur ';'
│   └── sample_submission.csv  # Format de soumission Kaggle (ID, TARGET)
├── notebooks/
│   ├── Project_Deep_ML.ipynb  # Approche ML classique : PCA + stacking
│   └── Project_Deep_DL.ipynb  # Approche Deep Learning : sélection L1 + MLP
├── presentation/
│   └── pres.pdf               # Slides de soutenance (ML vs DL)
├── _to_delete/                # Fichiers à supprimer (voir plus bas)
└── README.md
```

## Données

- **450 features** = 25 tâches d'écriture × 18 mesures par tâche (suffixe `1` à `25`) : `air_time`, `paper_time`, `total_time`, `disp_index`, `gmrt_in_air`, `gmrt_on_paper`, `mean_gmrt`, `max_x_extension`, `max_y_extension`, `mean_acc_*`, `mean_jerk_*`, `mean_speed_*`, `num_of_pendown`, `pressure_mean`, `pressure_var`.
- **Cible** : `class` — 65 patients (62,5 %) / 39 sains (37,5 %).
- Aucune valeur manquante. Très peu d'exemples pour beaucoup de variables → forte dimensionnalité.

## Approche 1 — Machine Learning classique (`Project_Deep_ML.ipynb`)

1. Analyse exploratoire (distribution des classes, corrélations, `SelectKBest`).
2. Split train/validation 80/20 stratifié.
3. Réduction de dimension : **PCA (80 composantes)** puis `StandardScaler`.
4. **StackingClassifier** avec XGBoost, LightGBM, CatBoost, ExtraTrees, RandomForest, Bagging (arbres) et SVM (RBF) ; méta-modèle CatBoost, `cv=10`.
5. Évaluation : **ROC-AUC validation ≈ 0,72–0,77** selon l'exécution.

## Approche 2 — Deep Learning (`Project_Deep_DL.ipynb`)

1. Même analyse exploratoire.
2. Rééquilibrage des classes avec **SMOTE**, puis split 80/20 et `StandardScaler`.
3. **Sélection de variables par réseau dense régularisé L1** : importance = poids absolus de la 1ʳᵉ couche → 45 meilleures features.
4. **MLP résiduel** (GELU, BatchNorm/LayerNorm, Dropout, bruit gaussien, petite branche Conv1D) entraîné avec :
   - **focal loss** (γ=2, α=0,75) + `class_weight` équilibré,
   - optimiseur **AdamW**, `EarlyStopping` et `ReduceLROnPlateau` sur `val_AUC`.
5. Évaluation : **ROC-AUC validation ≈ 0,86–0,98** selon l'exécution (meilleur run sauvegardé : AUC 0,976, accuracy 0,92 au seuil 0,5 et 0,96 au seuil optimal de Youden).

> ℹ️ Aucune graine TensorFlow n'est fixée et la validation ne contient que 26 exemples : les scores varient sensiblement d'une exécution à l'autre.

> ⚠️ Dans ce notebook, SMOTE est appliqué **avant** le split train/validation : des échantillons synthétiques dérivés de la validation se retrouvent dans l'entraînement, ce qui rend le score de validation optimiste. Pour une évaluation fiable, appliquer SMOTE uniquement sur le train (ou utiliser une validation croisée).

## Exécution

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow xgboost lightgbm catboost
cd notebooks
jupyter notebook
```

Lancer les notebooks depuis le dossier `notebooks/` (les chemins pointent vers `../data/`). Chaque notebook génère un fichier `submission.csv` (colonnes `ID`, `Target`) dans `notebooks/` : les deux notebooks écrivent le même nom de fichier, donc le dernier exécuté écrase l'autre.

## Fichiers à supprimer (`_to_delete/`)

| Fichier | Raison |
|---|---|
| `model with high performance.txt` | Brouillon : deux variantes copiées-collées du MLP résiduel (entrées PCA 80 et 100). Non utilisé par les notebooks ; le modèle final est dans `Project_Deep_DL.ipynb`. |
