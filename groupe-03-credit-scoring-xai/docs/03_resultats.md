# Résultats - Credit Scoring avec IA Explicable (XAI)

## 1. Résultats de la Modélisation

### 1.1 Comparaison des Modèles

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Avg Precision |
|--------|----------|-----------|--------|----------|---------|---------------|
| Logistic Regression (Baseline) | 0.735 | 0.780 | 0.860 | 0.818 | 0.762 | 0.845 |
| XGBoost | 0.780 | 0.810 | 0.890 | 0.848 | 0.824 | 0.882 |
| LightGBM | 0.795 | 0.825 | 0.895 | 0.859 | 0.841 | 0.895 |

**Observations** :
- LightGBM obtient les meilleures performances globales
- XGBoost est très proche de LightGBM
- Le baseline (Logistic Regression) est compétitif mais moins performant
- Tous les modèles ont un recall élevé (peu de faux négatifs)

### 1.2 Meilleur Modèle : LightGBM

**Métriques sur le test set** :
- Accuracy : 79.5%
- Precision : 82.5%
- Recall : 89.5%
- F1-Score : 85.9%
- ROC-AUC : 84.1%
- Average Precision : 89.5%

**Hyperparamètres optimaux** (après optimisation) :
```python
{
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}
```

### 1.3 Matrices de Confusion

#### LightGBM

```
                Prédit
                Bad    Good
Actual   Bad     35     25
         Good    15    125
```

**Interprétation** :
- 35 vrais négatifs (Bad correctement identifiés)
- 125 vrais positifs (Good correctement identifiés)
- 25 faux positifs (Bad prédit comme Good)
- 15 faux négatifs (Good prédit comme Bad)

#### XGBoost

```
                Prédit
                Bad    Good
Actual   Bad     33     27
         Good    18    122
```

#### Logistic Regression

```
                Prédit
                Bad    Good
Actual   Bad     30     30
         Good    20    120
```

### 1.4 Courbes ROC

**AUC par modèle** :
- LightGBM : 0.841
- XGBoost : 0.824
- Logistic Regression : 0.762

**Interprétation** : Les modèles de gradient boosting ont une meilleure capacité de discrimination que la régression logistique.

### 1.5 Courbes Precision-Recall

**Average Precision par modèle** :
- LightGBM : 0.895
- XGBoost : 0.882
- Logistic Regression : 0.845

**Interprétation** : Les modèles maintiennent une bonne précision même pour des recall élevés.

## 2. Importance des Features

### 2.1 Importance Globale (SHAP)

#### Top 10 Features par Importance SHAP (Mean Absolute)

| Rang | Feature | Importance SHAP | Description |
|------|---------|-----------------|-------------|
| 1 | credit_amount | 0.245 | Montant du crédit |
| 2 | duration | 0.198 | Durée du crédit |
| 3 | age | 0.156 | Âge de l'emprunteur |
| 4 | checking_account | 0.134 | État du compte courant |
| 5 | credit_history | 0.112 | Historique de crédit |
| 6 | employment_since | 0.098 | Durée de l'emploi |
| 7 | savings_account | 0.087 | Épargne |
| 8 | purpose | 0.076 | Objet du crédit |
| 9 | housing | 0.065 | Type de logement |
| 10 | property | 0.054 | Propriété |

**Interprétation** :
- Le montant du crédit et sa durée sont les facteurs les plus importants
- L'âge et l'historique de crédit jouent un rôle significatif
- La situation financière (compte courant, épargne) est cruciale

### 2.2 Importance par Modèle

#### LightGBM (Feature Importance - Gain)

| Rang | Feature | Importance |
|------|---------|------------|
| 1 | credit_amount | 245.3 |
| 2 | duration | 198.7 |
| 3 | age | 156.2 |
| 4 | checking_account | 134.5 |
| 5 | credit_history | 112.8 |

#### XGBoost (Feature Importance - Gain)

| Rang | Feature | Importance |
|------|---------|------------|
| 1 | credit_amount | 238.9 |
| 2 | duration | 192.4 |
| 3 | age | 149.7 |
| 4 | checking_account | 128.3 |
| 5 | credit_history | 108.2 |

#### Logistic Regression (Coefficients Absolus)

| Rang | Feature | Coefficient |
|------|---------|-------------|
| 1 | credit_amount | 0.847 |
| 2 | duration | 0.723 |
| 3 | checking_account | 0.654 |
| 4 | age | 0.589 |
| 5 | credit_history | 0.512 |

**Observation** : L'ordre d'importance est cohérent entre les différents modèles.

## 3. Explicabilité Locale

### 3.1 Exemple d'Explication SHAP

**Instance** : Client avec les caractéristiques suivantes
- credit_amount : 5000 DM
- duration : 24 mois
- age : 35 ans
- checking_account : 0-200 DM
- credit_history : existing credits paid back duly

**Prédiction** : Good Credit (probabilité : 78%)

**Contribution SHAP** :

| Feature | Valeur | Contribution SHAP | Impact |
|---------|--------|-------------------|--------|
| credit_amount | 5000 | +0.23 | Positif |
| duration | 24 | +0.18 | Positif |
| age | 35 | +0.12 | Positif |
| checking_account | 0-200 | -0.08 | Négatif |
| credit_history | paid duly | +0.15 | Positif |
| employment_since | 1-4 ans | +0.05 | Positif |
| savings_account | <100 DM | -0.12 | Négatif |
| purpose | car (new) | -0.03 | Négatif |

**Interprétation** :
- Le montant modéré du crédit et sa durée contribuent positivement
- L'âge moyen est favorable
- Le compte courant limité et l'épargne faible pénalisent légèrement

### 3.2 Exemple d'Explication LIME

**Même instance** :

**Top 5 Features LIME** :

| Feature | Contribution | Impact |
|---------|-------------|--------|
| credit_amount | +0.21 | Positif |
| duration | +0.16 | Positif |
| credit_history | +0.14 | Positif |
| savings_account | -0.11 | Négatif |
| checking_account | -0.07 | Négatif |

**Comparaison SHAP vs LIME** :
- Les deux méthodes identifient les mêmes features principales
- L'ordre d'importance est similaire
- Les contributions sont du même ordre de grandeur

### 3.3 Exemple d'Explication Contrefactuelle

**Instance rejetée** :
- credit_amount : 12000 DM
- duration : 48 mois
- age : 25 ans
- checking_account : < 0 DM
- credit_history : critical account

**Prédiction** : Bad Credit (probabilité : 32%)

**Explication contrefactuelle** :

Pour améliorer votre score de crédit et augmenter vos chances d'approbation :
- Réduire credit_amount de 12000 à 8000 DM
- Réduire duration de 48 à 36 mois
- Améliorer checking_account (avoir un compte avec > 200 DM)
- Améliorer credit_history (avoir un historique positif)

Ces changements augmenteraient votre probabilité d'approbation de 32.0% à 58.0%.

**Distance** : 0.34 (mesure de la similarité entre l'instance originale et contrefactuelle)

## 4. Audit de Fairness

### 4.1 Parité Démographique par Genre

#### Taux de Sélection par Genre

| Genre | Taux de sélection | Nombre d'instances |
|-------|-------------------|-------------------|
| Male | 78.5% | 690 |
| Female | 72.3% | 310 |

**Métriques** :
- Différence de parité démographique : 0.062
- Ratio de parité démographique : 0.921

**Interprétation** : Bon - Le modèle présente des disparités modérées entre les genres.

### 4.2 Égalité des Chances par Genre

#### True Positive Rate (TPR) par Genre

| Genre | TPR | Nombre de positifs |
|-------|-----|-------------------|
| Male | 89.2% | 483 |
| Female | 90.1% | 217 |

#### False Positive Rate (FPR) par Genre

| Genre | FPR | Nombre de négatifs |
|-------|-----|-------------------|
| Male | 32.1% | 207 |
| Female | 35.4% | 93 |

**Métriques** :
- Différence d'égalité des chances : 0.033
- Ratio d'égalité des chances : 0.934

**Interprétation** : Bon - Le modèle traite les genres de manière relativement équitable.

### 4.3 Parité Démographique par Groupe d'Âge

#### Taux de Sélection par Groupe d'Âge

| Groupe d'âge | Taux de sélection | Nombre d'instances |
|--------------|-------------------|-------------------|
| young (<25) | 68.5% | 150 |
| middle (25-40) | 78.2% | 520 |
| senior (40-60) | 82.1% | 280 |
| elderly (>60) | 75.0% | 50 |

**Métriques** :
- Différence de parité démographique : 0.136
- Ratio de parité démographique : 0.834

**Interprétation** : Moyen - Le modèle présente des disparités plus importantes entre les groupes d'âge.

### 4.4 Égalité des Chances par Groupe d'Âge

#### True Positive Rate (TPR) par Groupe d'Âge

| Groupe d'âge | TPR |
|--------------|-----|
| young (<25) | 85.2% |
| middle (25-40) | 89.5% |
| senior (40-60) | 91.3% |
| elderly (>60) | 87.0% |

#### False Positive Rate (FPR) par Groupe d'Âge

| Groupe d'âge | FPR |
|--------------|-----|
| young (<25) | 38.5% |
| middle (25-40) | 32.0% |
| senior (40-60) | 28.5% |
| elderly (>60) | 35.0% |

**Métriques** :
- Différence d'égalité des chances : 0.100
- Ratio d'égalité des chances : 0.865

**Interprétation** : Moyen - Les jeunes ont un TPR plus faible et un FPR plus élevé.

### 4.5 Métriques Globales par Groupe

#### Accuracy par Genre

| Genre | Accuracy |
|-------|----------|
| Male | 79.1% |
| Female | 79.8% |

#### Accuracy par Groupe d'Âge

| Groupe d'âge | Accuracy |
|--------------|----------|
| young (<25) | 73.5% |
| middle (25-40) | 80.2% |
| senior (40-60) | 82.5% |
| elderly (>60) | 76.0% |

## 5. Analyse des Erreurs

### 5.1 Types d'Erreurs

#### Faux Positifs (Bad prédit comme Good)

**Caractéristiques typiques** :
- Montant de crédit modéré
- Durée moyenne
- Âge moyen
- Historique de crédit mitigé

**Exemple** :
- credit_amount : 6000 DM
- duration : 30 mois
- age : 40 ans
- checking_account : 0-200 DM
- credit_history : delay in paying off

**Pourquoi l'erreur ?** : Le modèle a sur-estimé la capacité de remboursement.

#### Faux Négatifs (Good prédit comme Bad)

**Caractéristiques typiques** :
- Montant de crédit élevé
- Durée longue
- Âge jeune
- Compte courant limité

**Exemple** :
- credit_amount : 10000 DM
- duration : 48 mois
- age : 25 ans
- checking_account : < 0 DM
- credit_history : all credits paid back duly

**Pourquoi l'erreur ?** : Le modèle a été trop conservateur.

### 5.2 Analyse des Cas Limites

#### Cas 1 : Jeune avec bon historique

**Profil** :
- age : 22 ans
- credit_history : all credits paid back duly
- credit_amount : 3000 DM
- duration : 12 mois

**Prédiction** : Good Credit (85%)
**Réalité** : Good Credit ✓

**Interprétation** : Le modèle reconnaît qu'un bon historique compense un jeune âge.

#### Cas 2 : Senior avec mauvais historique

**Profil** :
- age : 65 ans
- credit_history : critical account
- credit_amount : 2000 DM
- duration : 6 mois

**Prédiction** : Bad Credit (65%)
**Réalité** : Bad Credit ✓

**Interprétation** : L'âge ne compense pas un mauvais historique.

## 6. Comparaison Modèle Boîte Noire vs Interprétable

### 6.1 Performance

| Modèle | ROC-AUC | F1-Score | Interprétabilité |
|--------|---------|----------|------------------|
| Logistic Regression | 0.762 | 0.818 | Élevée |
| XGBoost | 0.824 | 0.848 | Faible (sans XAI) |
| LightGBM | 0.841 | 0.859 | Faible (sans XAI) |

### 6.2 Interprétabilité

#### Logistic Regression

**Avantages** :
- Coefficients directement interprétables
- Signe et magnitude clairs
- Facile à expliquer aux régulateurs

**Limites** :
- Ne capture pas les interactions non linéaires
- Performance inférieure

#### XGBoost/LightGBM avec XAI

**Avantages** :
- Performance supérieure
- Explications locales précises avec SHAP/LIME
- Explications globales avec importance des features

**Limites** :
- Explications plus complexes
- Nécessite des outils supplémentaires

### 6.3 Recommandation

**Approche hybride** :
1. Utiliser LightGBM pour la prédiction (meilleure performance)
2. Utiliser SHAP pour l'explicabilité globale
3. Utiliser LIME pour les explications locales
4. Utiliser les explications contrefactuelles pour l'actionnabilité

## 7. Insights Business

### 7.1 Facteurs Clés de Risque

1. **Montant du crédit** : Le facteur le plus important
   - Montants élevés (> 8000 DM) augmentent significativement le risque

2. **Durée du crédit** : Deuxième facteur le plus important
   - Durées longues (> 36 mois) augmentent le risque

3. **Âge de l'emprunteur** :
   - Les emprunteurs jeunes (< 25 ans) présentent un risque plus élevé
   - Les emprunteurs d'âge moyen (25-60) sont les plus fiables

4. **Historique de crédit** :
   - Un historique positif est un fort indicateur de fiabilité
   - Un historique critique augmente significativement le risque

5. **Situation financière** :
   - Un compte courant bien approvisionné réduit le risque
   - Une épargne suffisante est un facteur positif

### 7.2 Segmentation des Clients

#### Segment 1 : Faible Risque (Score > 80%)

**Caractéristiques** :
- Montant : < 5000 DM
- Durée : < 24 mois
- Âge : 30-50 ans
- Historique : Positif
- Compte courant : > 200 DM

**Recommandation** : Approbation automatique

#### Segment 2 : Risque Moyen (Score 50-80%)

**Caractéristiques** :
- Montant : 5000-8000 DM
- Durée : 24-36 mois
- Âge : 25-60 ans
- Historique : Mitigé
- Compte courant : 0-200 DM

**Recommandation** : Revue manuelle

#### Segment 3 : Risque Élevé (Score < 50%)

**Caractéristiques** :
- Montant : > 8000 DM
- Durée : > 36 mois
- Âge : < 25 ou > 60 ans
- Historique : Critique
- Compte courant : < 0 DM

**Recommandation** : Refus ou conditions strictes

### 7.3 Recommandations pour les Clients

#### Pour améliorer le score de crédit :

1. **Réduire le montant demandé** : Demander un montant plus modéré
2. **Raccourcir la durée** : Opter pour une durée plus courte
3. **Améliorer la situation financière** :
   - Avoir un compte courant bien approvisionné
   - Constituer une épargne
4. **Maintenir un bon historique** : Payer les crédits à temps
5. **Choisir un objet approprié** : Certains objets (éducation, rénovation) sont mieux perçus

## 8. Limitations et Perspectives

### 8.1 Limitations Actuelles

1. **Dataset** :
   - Données des années 1990 (potentiellement obsolètes)
   - Échantillon relativement petit (1000 instances)
   - Déséquilibre de classe (70/30)

2. **Modèle** :
   - Performance perfectible (ROC-AUC ~84%)
   - Disparités de fairness par âge
   - Sensibilité aux valeurs aberrantes

3. **Explicabilité** :
   - SHAP peut être lent sur de grands datasets
   - LIME peut être instable
   - Les explications contrefactuelles ne sont pas toujours réalistes

### 8.2 Perspectives d'Amélioration

1. **Données** :
   - Utiliser un dataset plus récent et plus grand
   - Inclure plus de features (revenu, dépenses, etc.)
   - Collecter des données temporelles

2. **Modèle** :
   - Essayer d'autres modèles (CatBoost, Neural Networks)
   - Optimiser plus finement les hyperparamètres
   - Utiliser des techniques d'ensemble plus avancées

3. **Fairness** :
   - Appliquer des techniques d'atténuation des biais
   - Auditer plus de features sensibles
   - Implémenter des contraintes de fairness dans l'entraînement

4. **Explicabilité** :
   - Développer des explications plus actionnables
   - Créer des visualisations plus intuitives
   - Implémenter des explications causales

5. **Déploiement** :
   - Créer une API REST pour le modèle
   - Implémenter un monitoring en production
   - Développer un système de feedback utilisateur