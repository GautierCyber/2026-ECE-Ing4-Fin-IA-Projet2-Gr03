# Méthodologie - Credit Scoring avec IA Explicable (XAI)

## 1. Pipeline Global du Projet

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE CREDIT SCORING XAI                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. COLLECTE  │ -> │ 2. CHARGEMENT│ -> │ 3. NETTOYAGE │
│   DES DONNÉES│    │   DES DONNÉES│    │   & EDA      │
└──────────────┘    └──────────────┘    └──────────────┘
                                                      │
                                                      v
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 8. DASHBOARD │ <- │ 7. FAIRNESS  │ <- │ 4. PRÉTRAITEMENT│
│   STREAMLIT  │    │    AUDIT     │    │   DES DONNÉES │
└──────────────┘    └──────────────┘    └──────────────┘
                           │                     │
                           v                     v
                   ┌──────────────┐    ┌──────────────┐
                   │ 6. EXPLICABILITÉ│ <- │ 5. MODÉLISATION│
                   │   (SHAP/LIME/ │    │   (XGBoost/  │
                   │  CONTREFACTUEL)│   │   LightGBM)  │
                   └──────────────┘    └──────────────┘
```

## 2. Collecte et Chargement des Données

### 2.1 Source des Données

**Dataset** : German Credit Dataset (UCI Machine Learning Repository)

**URL** : https://archive.ics.uci.edu/ml/datasets/statlog+german+credit+data

### 2.2 Module `data_loader.py`

Le module `data_loader.py` implémente la classe `DataLoader` qui gère :

- **Téléchargement automatique** du dataset depuis l'UCI
- **Mapping des valeurs catégorielles** codes vers descriptions lisibles
- **Extraction des features sensibles** (genre, âge)
- **Création de groupes d'âge** pour l'audit de fairness
- **Transformation de la cible** (1=Good, 2=Bad → 1=Good, 0=Bad)

**Fonction principale** :
```python
def load_data(force_download: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Charge et prépare les données du German Credit Dataset.
    
    Returns:
        X: Features
        y: Target (0 = Bad Credit, 1 = Good Credit)
    """
```

### 2.3 Structure des Données

```python
# Features (X)
X = {
    # Features numériques
    'duration': int,           # Durée en mois
    'credit_amount': float,    # Montant en DM
    'installment_rate': int,   # Taux d'installments
    'residence_since': int,    # Années de résidence
    'age': int,                # Âge
    'existing_credits': int,   # Crédits existants
    'people_liable': int,      # Personnes responsables
    
    # Features catégorielles
    'checking_account': str,    # État du compte courant
    'credit_history': str,      # Historique de crédit
    'purpose': str,            # Objet du crédit
    'savings_account': str,     # Épargne
    'employment_since': str,    # Durée de l'emploi
    'personal_status': str,     # Statut personnel
    'other_debtors': str,       # Autres débiteurs
    'property': str,           # Propriété
    'other_installment_plans': str,  # Autres plans
    'housing': str,            # Logement
    'job': str,                # Niveau d'emploi
    'telephone': str,          # Téléphone
    'foreign_worker': str,     # Travailleur étranger
    
    # Features dérivées
    'gender': str,             # Genre (extrait de personal_status)
    'age_group': str           # Groupe d'âge
}

# Target (y)
y = {
    0: 'Bad Credit',   # Mauvais crédit
    1: 'Good Credit'   # Bon crédit
}
```

## 3. Analyse Exploratoire des Données (EDA)

### 3.1 Objectifs de l'EDA

- Comprendre la distribution des variables
- Identifier les valeurs manquantes et aberrantes
- Analyser les corrélations entre variables
- Détecter les déséquilibres de classe
- Identifier les patterns dans les données

### 3.2 Analyses Réalisées

#### Distribution de la Cible

```python
# Distribution du risque de crédit
Good Credit: 700 (70%)
Bad Credit: 300 (30%)
```

**Observation** : Déséquilibre de classe modéré (70/30)

#### Distribution des Variables Numériques

- **Duration** : Distribution asymétrique vers la droite (médiane ~18 mois)
- **Credit Amount** : Distribution asymétrique (médiane ~3000 DM)
- **Age** : Distribution approximativement normale (moyenne ~35 ans)

#### Analyse des Variables Catégorielles

- **Checking Account** : La plupart des clients n'ont pas de compte courant
- **Credit History** : Majorité de clients avec historique positif
- **Purpose** : Les crédits pour voiture et meubles sont les plus fréquents

#### Corrélations

- Forte corrélation positive entre duration et credit_amount
- Corrélation modérée entre age et credit_amount

## 4. Prétraitement des Données

### 4.1 Module `preprocessing.py`

Le module `preprocessing.py` implémente la classe `DataPreprocessor` qui gère :

- **Encodage des variables catégorielles** (Label Encoding)
- **Standardisation des variables numériques** (StandardScaler)
- **Split des données** (Train/Validation/Test)
- **Sauvegarde/Chargement** du préprocesseur

### 4.2 Étapes de Prétraitement

#### 4.2.1 Encodage des Variables Catégorielles

```python
# Label Encoding pour chaque variable catégorielle
for col in CATEGORICAL_COLUMNS:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
```

**Raison** : Les modèles XGBoost et LightGBM peuvent gérer directement les variables catégorielles encodées.

#### 4.2.2 Standardisation des Variables Numériques

```python
# StandardScaler pour les variables numériques
scaler = StandardScaler()
X[NUMERICAL_COLUMNS] = scaler.fit_transform(X[NUMERICAL_COLUMNS])
```

**Raison** : Les modèles de gradient boosting sont moins sensibles à l'échelle, mais la standardisation aide à la convergence.

#### 4.2.3 Split des Données

```python
# Split stratifié pour préserver la distribution de la cible
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y,
    test_size=0.2,      # 20% pour le test
    validation_size=0.1 # 10% pour la validation (du train restant)
)
```

**Distribution finale** :
- Train : 700 échantillons (70%)
- Validation : 100 échantillons (10%)
- Test : 200 échantillons (20%)

### 4.3 Gestion du Déséquilibre de Classe

**Approche** : Utilisation de `class_weight='balanced'` dans les modèles

```python
# Logistic Regression
model = LogisticRegression(class_weight='balanced')

# XGBoost
model = XGBClassifier(scale_pos_weight=ratio_negative/ratio_positive)
```

## 5. Modélisation

### 5.1 Modèles Implémentés

#### 5.1.1 Baseline : Logistic Regression

**Fichier** : `src/models/baseline_model.py`

**Caractéristiques** :
- Modèle linéaire interprétable
- Coefficients directement interprétables
- Rapide à entraîner

**Hyperparamètres par défaut** :
```python
{
    'random_state': 42,
    'max_iter': 1000,
    'class_weight': 'balanced'
}
```

#### 5.1.2 XGBoost

**Fichier** : `src/models/xgboost_model.py`

**Caractéristiques** :
- Gradient boosting optimisé
- Gestion des valeurs manquantes
- Régularisation L1/L2
- Parallélisation

**Hyperparamètres par défaut** :
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss'
}
```

**Optimisation des hyperparamètres** :
```python
# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}
```

#### 5.1.3 LightGBM

**Fichier** : `src/models/lightgbm_model.py`

**Caractéristiques** :
- Leaf-wise tree growth (vs level-wise)
- Plus rapide et moins gourmand en mémoire
- Gestion optimisée des grandes datasets

**Hyperparamètres par défaut** :
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}
```

### 5.2 Entraînement des Modèles

#### 5.2.1 Processus d'Entraînement

```python
# 1. Initialiser le modèle
model = XGBoostModel()

# 2. Entraîner avec validation set
metrics = model.train(
    X_train, y_train,
    X_val, y_val,
    early_stopping_rounds=10
)

# 3. Évaluer sur le test set
test_metrics = model.evaluate(X_test, y_test)

# 4. Sauvegarder le modèle
model.save('data/models/xgboost_model.pkl')
```

#### 5.2.2 Métriques d'Évaluation

| Métrique | Description | Formule |
|----------|-------------|---------|
| Accuracy | Proportion de prédictions correctes | (TP + TN) / (TP + TN + FP + FN) |
| Precision | Proportion de positifs corrects parmi les prédits positifs | TP / (TP + FP) |
| Recall | Proportion de positifs corrects parmi les vrais positifs | TP / (TP + FN) |
| F1-Score | Moyenne harmonique de precision et recall | 2 × (Precision × Recall) / (Precision + Recall) |
| ROC-AUC | Aire sous la courbe ROC | - |
| Average Precision | Aire sous la courbe Precision-Recall | - |

### 5.3 Comparaison des Modèles

Le module `evaluation.py` implémente la classe `ModelEvaluator` pour comparer les modèles.

```python
evaluator = ModelEvaluator()
comparison = evaluator.compare_models(
    models={'Baseline': baseline, 'XGBoost': xgb, 'LightGBM': lgb},
    X_test=X_test,
    y_test=y_test
)
```

## 6. Explicabilité

### 6.1 SHAP (SHapley Additive exPlanations)

**Fichier** : `src/explainability/shap_explainer.py`

#### 6.1.1 Initialisation

```python
explainer = SHAPExplainer(model, feature_names=feature_names)
explainer.fit(X_train, explainer_type='tree')
```

#### 6.1.2 Explication Globale

```python
# Calculer les valeurs SHAP
shap_values = explainer.explain(X_test)

# Importance des features
importance = explainer.get_feature_importance(importance_type='mean_abs')
```

#### 6.1.3 Explication Locale

```python
# Explication pour une instance spécifique
local_expl = explainer.get_local_explanation(instance_idx, X_test)
```

#### 6.1.4 Visualisations

- **Summary Plot** : Vue d'ensemble de l'importance et de l'impact
- **Waterfall Plot** : Explication détaillée d'une prédiction
- **Force Plot** : Visualisation interactive
- **Dependence Plot** : Relation feature-contribution

### 6.2 LIME (Local Interpretable Model-agnostic Explanations)

**Fichier** : `src/explainability/lime_explainer.py`

#### 6.2.1 Initialisation

```python
explainer = LIMEExplainer(model, feature_names=feature_names)
explainer.fit(X_train)
```

#### 6.2.2 Explication Locale

```python
# Expliquer une instance
explanation = explainer.explain_instance(instance.values)

# Explication structurée
local_expl = explainer.get_local_explanation(instance.values, explanation)
```

#### 6.2.3 Comparaison SHAP vs LIME

```python
comparison = explainer.compare_with_shap(
    instance=instance.values,
    shap_explainer=shap_explainer,
    X=X_test,
    instance_idx=instance_idx
)
```

### 6.3 Explications Contrefactuelles

**Fichier** : `src/explainability/counterfactual.py`

#### 6.3.1 Génération

```python
explainer = CounterfactualExplainer(model, feature_names=feature_names)

# Générer une explication contrefactuelle
cf = explainer.generate_counterfactual(
    instance=instance,
    target_class=1,  # Bon crédit
    max_iterations=1000,
    learning_rate=0.1
)
```

#### 6.3.2 Explication Textuelle

```python
explanation = explainer.explain_counterfactual(cf)
print(explanation)
```

**Exemple de sortie** :
```
Pour améliorer votre score de crédit et augmenter vos chances d'approbation :
- Réduire credit_amount
- Augmenter duration
- Modifier checking_account

Ces changements augmenteraient votre probabilité d'approbation de 35.0% à 55.0%.
```

## 7. Audit de Fairness

**Fichier** : `src/fairness/fairness_audit.py`

### 7.1 Initialisation

```python
auditor = FairnessAuditor(model)
```

### 7.2 Audit de Parité Démographique

```python
results = auditor.audit_demographic_parity(
    X=X_test,
    y=y_test,
    sensitive_feature=X_test['gender']
)
```

**Métriques** :
- **Difference** : Différence maximale de taux de sélection entre groupes
- **Ratio** : Ratio minimum/maximum de taux de sélection

### 7.3 Audit d'Égalité des Chances

```python
results = auditor.audit_equalized_odds(
    X=X_test,
    y=y_test,
    sensitive_feature=X_test['gender']
)
```

**Métriques** :
- **Difference** : Différence maximale de TPR et FPR entre groupes
- **Ratio** : Ratio minimum/maximum de TPR et FPR

### 7.4 Audit Complet

```python
results = auditor.audit_all(
    X=X_test,
    y=y_test,
    sensitive_features=X_test[['gender', 'age_group']]
)
```

### 7.5 Atténuation des Biais

```python
mitigation = auditor.mitigate_fairness(
    X_train=X_train,
    y_train=y_train,
    sensitive_features_train=X_train['gender'],
    constraint='demographic_parity',
    method='exponentiated_gradient'
)
```

## 8. Dashboard Streamlit

**Fichier** : `src/dashboard/app.py`

### 8.1 Architecture

```
Dashboard Streamlit
├── Page Accueil
│   ├── Contexte du projet
│   ├── Statistiques des données
│   └── Distribution des features
├── Page Prédiction
│   ├── Sélection du modèle
│   ├── Sélection de l'instance
│   └── Affichage des résultats
├── Page Explicabilité
│   ├── Onglet SHAP
│   ├── Onglet LIME
│   └── Onglet Contrefactuel
├── Page Fairness
│   ├── Parité démographique
│   └── Égalité des chances
└── Page Comparaison Modèles
    ├── Tableau de comparaison
    └── Graphiques comparatifs
```

### 8.2 Lancement du Dashboard

```bash
streamlit run src/dashboard/app.py
```

### 8.3 Fonctionnalités

- **Prédiction interactive** : Sélection d'instances ou entrée manuelle
- **Explications visuelles** : Graphiques SHAP, LIME, contrefactuels
- **Audit de fairness** : Visualisation des disparités par groupe
- **Comparaison de modèles** : Tableaux et graphiques comparatifs

## 9. Organisation du Code

### 9.1 Structure des Modules

```
src/
├── __init__.py
├── config.py              # Configuration globale
├── data_loader.py         # Chargement des données
├── preprocessing.py       # Prétraitement
├── evaluation.py          # Évaluation des modèles
├── models/                # Modèles
│   ├── __init__.py
│   ├── baseline_model.py
│   ├── xgboost_model.py
│   └── lightgbm_model.py
├── explainability/        # Explicabilité
│   ├── __init__.py
│   ├── shap_explainer.py
│   ├── lime_explainer.py
│   └── counterfactual.py
├── fairness/              # Fairness
│   ├── __init__.py
│   └── fairness_audit.py
└── dashboard/             # Dashboard
    ├── __init__.py
    └── app.py
```

### 9.2 Bonnes Pratiques

- **Documentation** : Docstrings pour toutes les fonctions et classes
- **Logging** : Utilisation du module logging pour le suivi
- **Type Hints** : Annotations de type pour la clarté
- **Tests** : Tests unitaires dans le dossier `tests/`
- **Configuration** : Paramètres centralisés dans `config.py`

## 10. Workflow de Développement

### 10.1 Environnement de Développement

```bash
# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 10.2 Exécution des Notebooks

```bash
# Lancer Jupyter
jupyter notebook

# Exécuter les notebooks dans l'ordre
# 1. 01_eda.ipynb - Analyse exploratoire
# 2. 02_modeling.ipynb - Modélisation
# 3. 03_explainability.ipynb - Explicabilité
# 4. 04_fairness.ipynb - Fairness
```

### 10.3 Tests

```bash
# Exécuter les tests unitaires
pytest tests/

# Avec couverture de code
pytest tests/ --cov=src --cov-report=html
```

## 11. Déploiement

### 11.1 Sauvegarde des Modèles

```python
# Sauvegarder un modèle
model.save('data/models/xgboost_model.pkl')

# Charger un modèle
model = XGBoostModel()
model.load('data/models/xgboost_model.pkl')
```

### 11.2 Export du Dashboard

```bash
# Créer un exécutable standalone
pip install pyinstaller
pyinstaller --onefile src/dashboard/app.py
```

## 12. Références Techniques

### 12.1 Documentation des Bibliothèques

- **XGBoost** : https://xgboost.readthedocs.io/
- **LightGBM** : https://lightgbm.readthedocs.io/
- **SHAP** : https://shap.readthedocs.io/
- **LIME** : https://lime-ml.readthedocs.io/
- **Fairlearn** : https://fairlearn.org/
- **Streamlit** : https://docs.streamlit.io/

### 12.2 Standards et Bonnes Pratiques

- **PEP 8** : Style Guide for Python Code
- **PEP 257** : Docstring Conventions
- **Scikit-learn API** : https://scikit-learn.org/developers/develop.html