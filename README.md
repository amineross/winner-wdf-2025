# Winners of West Data Festival AI Challenge (ESIEA x LUMINESS) 🥇

Ce git contient les solutions développées par **Amine ROSTANE**, **Gaël GARNIER**, et **Augustin LOGEAIS** dans le cadre du **West Data Festival AI Challenge** organisé par **ESIEA x LUMINESS**.

Notre approche hybride combinant **MILP**, **recuit simulé**, et **heuristiques gloutonnes** nous a permis d’obtenir un **score 48 % supérieur** aux scores de référence du benchmark officiel.

---

## 📁 Fichiers du dépôt

| Fichier | Description |
|--------|-------------|
| `greedy_simulated_annealing_4_48.py` | Recuit simulé + heuristique gloutonne, fonctionne pour les instances `4_*.json` à `48_*.json`. |
| `greedy_simulated_annealing_75_100.py` | Recuit simulé avancé pour les grosses instances `75_*.json` à `100_*.json`. |
| `hybrid_milp_greedy_simulated_annealing_4_48.py` | Combine MILP local + recuit simulé + heuristique pour les instances `4_*.json` à `48_*.json`. |
| `hybrid_milp_greedy_simulated_annealing_75_100.py` | Approche hybride MILP + métaheuristique pour les instances `75_*.json` à `100_*.json`. |
| `milp_exact.py` | Résolution exacte du problème par MILP pur (⚠️ très lent au-delà de `4_*.json`). |
| `simplified_milp.py` | Version plus légère du solveur MILP, utilisable sur petites instances. |
| `script.py` | Contient la définition de la classe `Instance`, les contraintes et les fonctions de simulation. |

---

## ⚙️ Installation

Ce projet utilise **Python 3.9+** et les dépendances suivantes :

```bash
pip install pulp networkx numpy

```

> 📌 **Dépendance principale** : [`pulp`](https://pypi.org/project/PuLP/) est utilisée pour la modélisation MILP.

----------

## 💻 Exécution

Les scripts sont configurés pour exécuter les instances correspondant à leur nom.

Par exemple :

```bash
python hybrid_milp_greedy_simulated_annealing_4_48.py

```

Exécutera automatiquement tous les fichiers `4_*.json` à `48_*.json` présents dans le répertoire courant.

Pour **changer les instances exécutées**, il suffit de **modifier le pattern `glob.glob(...)`** dans la fonction `main()` en fin de fichier.

----------

## 🚀 Performances et remarques

-   Certains algorithmes ne fonctionnent que sur une plage d’instances donnée (ex: `hybrid_milp_*` fonctionne mal au-delà de `48_*.json`).
    
-   Les méthodes **MILP** sont très performantes mais **lentes** sur les moyennes et grosses instances.
    
-   Nous avons utilisé des **machines virtuelles Google Cloud C2 standard** avec le solveur **IBM CPLEX** pour accélérer les résolutions.
    

----------

## 🧠 Utiliser IBM CPLEX au lieu de PULP (optionnel)

Nos codes utilisent `pulp.PULP_CBC_CMD()` par défaut, mais vous pouvez activer **CPLEX** comme suit :

1.  Installer IBM CPLEX (gratuit pour usage académique) :
    
    -   [Guide officiel d'installation](https://www.ibm.com/docs/en/icos/20.1.0?topic=python-setting-up-cplex-optimization-studio)
        
2.  Modifier la ligne suivante dans les fonctions MILP de nos fichiers :
    

```python
solver = pulp.PULP_CBC_CMD(timeLimit=self.milp_time_limit, msg=0)

```

**Remplacez-la par :**

```python
solver = pulp.CPLEX_CMD(timeLimit=self.milp_time_limit, msg=0)

```

> 📎 Cette ligne se trouve dans la méthode `optimize_time_window_milp(...)`.

----------

## 📫 Contact

Pour toute question ou suggestion, vous pouvez nous contacter :

-   Amine ROSTANE — [LinkedIn](https://www.linkedin.com/in/amine-rostane-742406266/)
    
-   Gaël GARNIER — [LinkedIn](https://www.linkedin.com/in/ga%C3%ABlgarnier/)
    
-   Augustin LOGEAIS — [LinkedIn](https://www.linkedin.com/in/augustin-logeais-432891269/)
    

----------

## 🏁 Licence

Projet académique - pas de licence commerciale. Merci de nous contacter pour toute réutilisation étendue.

