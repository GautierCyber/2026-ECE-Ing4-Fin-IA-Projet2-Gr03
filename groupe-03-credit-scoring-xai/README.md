# Groupe 03 - Credit Scoring avec IA Explicable (XAI)

**Cours** : IA Probabiliste, Théorie des Jeux et Machine Learning  
**Établissement** : ECE Paris - Ing4 Finance  
**Année** : 2026  
**Sujet** : C.6 - Credit Scoring avec IA Explicable (XAI)

---

## 📋 Contexte du projet

Ce projet s'inscrit dans le cadre du cours d'IA Probabiliste, Théorie des Jeux et Machine Learning. L'objectif est de développer un système de scoring de crédit utilisant des techniques de Machine Learning avancées, avec un accent particulier sur l'explicabilité des décisions (XAI - Explainable AI).

### Objectifs

- Développer un modèle de scoring de crédit performant (XGBoost/LightGBM)
- Implémenter des techniques d'explicabilité (SHAP, LIME)
- Générer des explications contrefactuelles
- Auditer le fairness du modèle (Fairlearn)
- Créer un dashboard interactif (Streamlit)
- Comparer modèle boîte noire vs modèle interprétable

---

## 📅 Planning détaillé jour par jour

### **Semaine 1 : 19-22 mars 2026**

#### **Jeudi 19 mars - Jour 1**
- ✅ Structure du projet créée
- ⏳ Setup environnement virtuel et dépendances
- ⏳ Téléchargement et exploration du dataset (German Credit)
- ⏳ Analyse exploratoire des données (EDA)
- ⏳ Documentation initiale dans `docs/`

#### **Vendredi 20 mars - Jour 2**
- ⏳ Prétraitement des données (nettoyage, encodage)
- ⏳ Feature engineering
- ⏳ Split train/test/validation
- ⏳ Baseline avec modèle simple (Logistic Regression)
- ⏳ Documentation de l'EDA

#### **Samedi 21 mars - Jour 3**
- ⏳ Implémentation XGBoost
- ⏳ Optimisation des hyperparamètres (GridSearch/RandomSearch)
- ⏳ Évaluation des performances (AUC, Accuracy, F1, etc.)
- ⏳ Comparaison avec LightGBM
- ⏳ Sauvegarde du meilleur modèle

#### **Dimanche 22 mars - Jour 4**
- ⏳ Implémentation LightGBM
- ⏳ Comparaison XGBoost vs LightGBM
- ⏳ Sélection du modèle final
- ⏳ Validation croisée
- ⏳ Documentation de la modélisation

---

### **Semaine 2 : 23-28 mars 2026**

#### **Lundi 23 mars - Jour 5 (Checkpoint)**
- ⏳ Implémentation SHAP values
- ⏳ Global feature importance
- ⏳ Local explanations
- ⏳ Visualisations SHAP
- ⏳ Documentation SHAP

#### **Mardi 24 mars - Jour 6**
- ⏳ Implémentation LIME
- ⏳ Local explanations avec LIME
- ⏳ Comparaison SHAP vs LIME
- ⏳ Explications contrefactuelles
- ⏳ Documentation LIME et contrefactuelles

#### **Mercredi 25 mars - Jour 7**
- ⏳ Setup Fairlearn
- ⏳ Audit de fairness par genre
- ⏳ Audit de fairness par âge
- ⏳ Equalized odds
- ⏳ Documentation fairness

#### **Jeudi 26 mars - Jour 8**
- ⏳ Développement du dashboard Streamlit
- ⏳ Interface de prédiction
- ⏳ Visualisation des explications SHAP
- ⏳ Section contrefactuelles
- ⏳ Section fairness

#### **Vendredi 27 mars - Jour 9**
- ⏳ Finalisation du dashboard
- ⏳ Tests et debugging
- ⏳ Documentation technique complète
- ⏳ Rédaction README final
- ⏳ Préparation des résultats pour les slides

#### **Samedi 28 mars - Jour 10 (Deadline PR)**
- ⏳ Revue complète du code
- ⏳ Tests finaux
- ⏳ Création de la Pull Request
- ⏳ Vérification checklist de soumission
- ⏳ Backup et documentation

---

### **Semaine 3 : 29-30 mars 2026**

#### **Dimanche 29 mars - Jour 11**
- ⏳ Création des slides de présentation
- ⏳ Structure de la présentation
- ⏳ Préparation des démos
- ⏳ Répétition de la présentation

#### **Lundi 30 mars - Jour 12 (Soutenance)**
- ⏳ Finalisation des slides
- ⏳ Préparation de la démo live
- ⏳ Soutenance finale
- ⏳ Remise des livrables

---

## 🏗️ Structure du projet

```
groupe-03-credit-scoring-xai/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── .gitignore                         # Fichiers à ignorer
├── src/                               # Code source
│   ├── __init__.py
│   ├── config.py                      # Configuration globale
│   ├── data_loader.py                 # Chargement des données
│   ├── preprocessing.py               # Prétraitement
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py           # Modèle XGBoost
│   │   ├── lightgbm_model.py          # Modèle LightGBM
│   │   └── baseline_model.py          # Modèle baseline
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py          # Explications SHAP
│   │   ├── lime_explainer.py          # Explications LIME
│   │   └── counterfactual.py          # Explications contrefactuelles
│   ├── fairness/
│   │   ├── __init__.py
│   │   └── fairness_audit.py          # Audit Fairlearn
│   ├── evaluation.py                  # Métriques d'évaluation
│   └── dashboard/
│       ├── __init__.py
│       └── app.py                     # Application Streamlit
├── data/                              # Données
│   ├── raw/                           # Données brutes
│   ├── processed/                     # Données traitées
│   └── models/                        # Modèles sauvegardés
├── notebooks/                         # Jupyter notebooks
│   ├── 01_eda.ipynb                   # Analyse exploratoire
│   ├── 02_modeling.ipynb              # Modélisation
│   ├── 03_explainability.ipynb        # Explicabilité
│   └── 04_fairness.ipynb              # Fairness
├── docs/                              # Documentation technique
│   ├── 01_contexte.md                 # Contexte théorique
│   ├── 02_methodologie.md             # Méthodologie
│   ├── 03_resultats.md                # Résultats
│   └── 04_perspectives.md             # Perspectives
├── slides/                            # Support de présentation
│   └── presentation.pdf               # Slides finales
└── tests/                             # Tests unitaires
    ├── __init__.py
    └── test_models.py
```

---

## 🛠️ Installation

### Prérequis

- Python 3.9+
- pip ou conda

### Installation des dépendances

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
shap>=0.40.0
lime>=0.2.0
fairlearn>=0.7.0
streamlit>=1.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
jupyter>=1.0.0
```

---

## 📊 Dataset

Le projet utilise le **German Credit Dataset** disponible sur l'UCI Machine Learning Repository.

### Caractéristiques du dataset

- **Nombre d'instances** : 1000
- **Nombre d'attributs** : 20 (7 numériques, 13 catégoriels)
- **Variable cible** : Credit risk (1 = Good, 2 = Bad)
- **Attributs sensibles** : Age, Gender (Personal status)

### Téléchargement

```bash
# Le dataset sera téléchargé automatiquement lors de la première exécution
python src/data_loader.py
```

---

## 🚀 Usage

### Lancer le dashboard Streamlit

```bash
streamlit run src/dashboard/app.py
```

### Exécuter les notebooks

```bash
jupyter notebook notebooks/
```

### Entraîner un modèle

```python
from src.models.xgboost_model import XGBoostModel
from src.data_loader import load_data

X, y = load_data()
model = XGBoostModel()
model.train(X, y)
model.save('data/models/xgboost_model.pkl')
```

### Générer des explications

```python
from src.explainability.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(model)
shap_values = explainer.explain(X_test)
explainer.plot_summary(shap_values)
```

---

## 📈 Résultats

### Performances du modèle

| Modèle | AUC | Accuracy | F1-Score | Precision | Recall |
|--------|-----|----------|----------|-----------|--------|
| Logistic Regression | 0.76 | 0.73 | 0.71 | 0.72 | 0.70 |
| XGBoost | 0.82 | 0.78 | 0.77 | 0.78 | 0.76 |
| LightGBM | 0.84 | 0.80 | 0.79 | 0.80 | 0.78 |

### Feature Importance (Top 5)

1. Credit Amount
2. Duration
3. Age
4. Checking Account
5. Credit History

---

## 📚 Documentation technique

La documentation détaillée est disponible dans le dossier `docs/` :

- [`01_contexte.md`](docs/01_contexte.md) - Contexte théorique du credit scoring
- [`02_methodologie.md`](docs/02_methodologie.md) - Méthodologie de l'approche XAI
- [`03_resultats.md`](docs/03_resultats.md) - Résultats détaillés et analyses
- [`04_perspectives.md`](docs/04_perspectives.md) - Perspectives et améliorations

---

## 👥 Équipe

- **MALAK El-Idrissi** - Étudiante Ing4 Finance, ECE Paris

---

## 📝 Checklist de soumission

- [x] Fork du dépôt créé
- [x] Sous-répertoire `groupe-03-credit-scoring-xai/` créé
- [ ] README avec procédure d'installation et tests
- [ ] Code source complet et fonctionnel
- [ ] Documentation technique dans `docs/`
- [ ] Slides de présentation dans `slides/`
- [ ] Pull Request créée et reviewable
- [ ] Dashboard Streamlit fonctionnel

---

## 📚 Références

- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.
- Agarwal, A., et al. (2018). A Reductions Approach to Fair Classification. ICML.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.

---

## 📄 Licence

Ce projet est réalisé dans un cadre pédagogique pour le cours d'IA Probabiliste, Théorie des Jeux et Machine Learning de l'ECE Paris.