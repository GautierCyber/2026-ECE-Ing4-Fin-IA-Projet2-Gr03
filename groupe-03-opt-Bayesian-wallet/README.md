# **C.5 - Optimisation de Portefeuille Bayesien (Black-Litterman)**

**Difficulté** : 3/5 | **Domaine** : Probabilités, Machine Learning

---

## **📌 Description**

Au-delà de la théorie de Markowitz classique, ce projet implémente le **modèle Black-Litterman** pour intégrer des *"views"* (opinions) probabilistes sur les rendements futurs. L’approche bayésienne combine un *prior* (équilibre de marche) avec des *views* de l’investisseur pour obtenir une allocation plus stable et intuitive. Ce modèle est largement utilisé par les asset managers institutionnels.

---

## **🎯 Objectifs Gradés**


| Niveau        | Objectifs                                                                                                           | Statut |
| ------------- | ------------------------------------------------------------------------------------------------------------------- | ------ |
| **Minimum**   | Implémentation Black-Litterman avec *views* simples, comparaison avec Markowitz classique sur données Yahoo Finance | 🟡     |
| **Bon**       | *Views* avec niveaux de confiance variables, optimisation sous contraintes (budget, secteur), frontière efficiente  | 🟡     |
| **Excellent** | *Views* générées par ML (sentiment, momentum), backtesting multi-périodes, analyse de sensibilité aux *views*       | 🟡     |


---

## **📚 Notebooks de Référence**


| Notebook  | Description                              | Lien      |
| --------- | ---------------------------------------- | --------- |
| Infer-101 | Inference bayésienne (prior + posterior) | [Lien](#) |
| QC-Py-21  | Optimisation de portefeuille ML          | [Lien](#) |


---

## **🔗 Références Externes**


| Source                            | Description                                                   | Lien                                                                       |
| --------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------- |
| PyPortfolioOpt - Black-Litterman  | Implémentation directe en Python (point de départ recommandé) | [Lien](https://github.com/robertmartin8/PyPortfolioOpt)                    |
| Riskfolio-Lib                     | Optimisation avancée avec Black-Litterman                     | [Lien](https://riskfolio-lib.readthedocs.io/)                              |
| Black-Litterman Model (Wikipedia) | Référence théorique                                           | [Lien](https://en.wikipedia.org/wiki/Black%E2%80%93Litterman_model)        |
| Thomas Starke - BL Model          | Tutoriels financiers Python                                   | [Lien](https://www.quantconnect.com/learn/tutorials/black-litterman-model) |


---

## **🛠️ Installation**

### **Prérequis**

- Python 3.8+
- `pip` ou `conda`

### **Étapes**

1. **Cloner le dépôt** :
  ```bash
   git clone https://github.com/TON_NOM_D_UTILISATEUR/nom-du-depot.git
   cd nom-du-depot
  ```
2. **Créer un environnement virtuel** (recommandé) :
  ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
  ```
3. **Installer les dépendances** :
  ```bash
   pip install -r requirements.txt
  ```
   *Si tu n’as pas de `requirements.txt`, voici les packages essentiels :*

---

## **📁 Structure du Projet**

```
nom-du-depot/
├── README.md                # Ce fichier
├── requirements.txt         # Dépendances Python
├── notebooks/               # Notebooks Jupyter
│   ├── Infer-101.ipynb      # Inference bayésienne
│   └── QC-Py-21.ipynb       # Optimisation de portefeuille
├── src/                     # Code source
│   ├── black_litterman.py   # Implémentation du modèle
│   └── utils.py             # Fonctions utilitaires
└── tests/                   # Tests unitaires et d'intégration
    ├── test_black_litterman.py
    └── test_utils.py
```

---

## **🧪 Exécuter les Tests**

1. **Installer `pytest**` (si ce n’est pas déjà fait) :
  ```bash
   pip install pytest
  ```
2. **Lancer les tests** :
  ```bash
   pytest tests/
  ```
3. **Résultats attendus** :
  - Tous les tests doivent passer (✅).
  - Si un test échoue, vérifie les messages d’erreur et corrige le code.

---

## **🚀 Exécuter les Notebooks**

1. **Lancer Jupyter Lab** :
  ```bash
   jupyter lab
  ```
2. **Ouvrir les notebooks** dans le dossier `notebooks/` et exécuter les cellules dans l’ordre.

---

## **📝 Contribuer**

- Ouvre une **Pull Request** depuis une branche dédiée.
- Respecte le [Code de Conduite](CODE_OF_CONDUCT.md) (à créer si nécessaire).
- Documente tes modifications dans le code et les notebooks.

---

**💡 Astuce** : Pour les *views* avancées (niveau "Excellent"), explore les bibliothèques comme `TA-Lib` pour le *momentum* ou `TextBlob` pour le *sentiment analysis*.

---

*Dernière mise à jour : 17/03/2026*