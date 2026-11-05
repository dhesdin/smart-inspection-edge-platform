# Journal de bord - Smart Inspection Edge Platform

## Date : conception initiale, choix méthodologiques et bootstrap technique

_Début juillet_

### 1. Contexte

Démarrage du projet Smart Inspection Edge Platform : passage du cadrage théorique (dataset, méthodes, architecture) à l'initialisation technique complète du repo.

### 2. Ce qui a été fait concrètement

Choix et téléchargement du dataset MVTec AD (source officielle, licence CC BY-NC-SA vérifiée)

Choix des méthodes : PaDiM et STFPM, toutes deux from scratch, benchmarkées l'une contre l'autre

Conception de l'architecture src/smart_inspection/ : data/, models/ / sous-packages padim - stfpm, training/, evaluation/, api/, config/, io/

Séparation des fichiers de configuration : configs/common.yaml, configs/padim.yaml, configs/stfpm.yaml

Bootstrap complet du repo : git init, remote SSH, README.md, .gitignore, LICENSE (MIT)

Écriture de pyproject.toml (dépendances, groupes dev/edge(phase2), config ruff/pytest, src-layout)

Écriture du Makefile (setup, test, lint, clean)

Installation et vérification de l'environnement sur deux machines : laptop (PyTorch CPU) et PC fixe (PyTorch 2.13.0, RTX 3070 Ti, CUDA 13.2 confirmés fonctionnels)

Structure outputs/ (models, figures, reports, logs) avec .gitkeep

### 3. Difficultés rencontrées et comment elles ont été résolues

#### Rôle des masques ground_truth/ pour les images good/ du test set

**Problème :** je pensais qu'il fallait "superposer" les masques et entraîner le modèle dessus.

**Fausse hypothèse :** je mélangeais phase d'entraînement et phase d'évaluation.

**Insight :** les masques ne sont utilisés qu'après coup, uniquement pour calculer une métrique (comparaison heatmap générée vs masque réel), jamais vus par le modèle. Pour les images good/ (qui n'ont pas de masque physique sur le disque), il faut générer un masque vide en code, seulement pour l'évaluation.

#### Erreur de syntaxe Makefile (ifeq et tuple Python)

**Problème 1 :** if eq (...) au lieu de ifeq (...) - syntaxe Make invalide.

**Problème 2 :** sys.version_info < (3.10) - (3.10) n'est pas un tuple sans virgule, donc la comparaison ne fonctionnait pas comme prévu.

**Résolution :** ifeq en un mot, et (3,10) avec virgule pour créer un vrai tuple.

#### NVML library version mismatch sur le PC fixe

**Problème :** nvidia-smi échouait après installation des dépendances CUDA via pip.

**Résolution :** reboot machine -> fonctionne.

### 4. Décisions d'architecture / conception, avec le raisonnement

Strategy Pattern pour models/ (classe abstraite AnomalyMethod + factory.py) plutôt que des if/else dispersés : permet à training/ et au futur api/ d'appeler method.fit()/method.predict() sans jamais besoin de connaître l'implémentation sous-jacente (STFPM - PaDiM).

api/ comme sous-package dans src/ plutôt qu'à la racine du repo : un seul artefact installable (pip install .) + Docker en un seul bloc.

Trois fichiers YAML (common.yaml + padim.yaml + stfpm.yaml) plutôt qu'un seul fichier avec sections : permet de charger uniquement la sous-config pertinente selon la méthode entraînée.

PaDiM ET STFPM from scratch, plutôt qu'utiliser Anomalib pour l'une des deux : décision prise après avoir chiffré le coût en temps : j'ai jugé que l'apprentissage des statistiques multivariées (mean/covariance, Mahalanobis) est transférable au-delà de ce seul projet, contrairement à l'usage d'une librairie toute faite.

Contraintes de version souples avec borne haute (torch>=2.13,<3.0) dans pyproject.toml, plutôt que versions figées : flexibilité sur les correctifs mineurs, protection contre une rupture de version majeure non testée. Un lock file séparé (à générer via pip freeze) reste la source de vérité pour la reproductibilité stricte.

PyTorch installé différemment selon la machine (CPU sur laptop, CUDA 13.2 sur PC fixe) sans que pyproject.toml ne distingue les deux : le choix de variante matérielle se fait à l'installation, documenté dans le README, pas dans le fichier de dépendances du projet.

### 5. Ce que j'ai appris

J'ai dû sortir du réflexe classification supervisée que j'avais avec CIFAR-10 pour comprendre qu'en anomaly detection one-class, il n'y a structurellement aucun label pendant l'entraînement - soit on fait de la reconstruction, soit de la distillation teacher/student, mais jamais de cross-entropy classique sur une seule classe.

J'ai appris à séparer clairement ce qui sert à l'entraînement de ce qui sert uniquement à l'évaluation - typiquement les masques de ground truth ne sont jamais vus par le modèle, ils servent seulement à calculer une métrique après coup.

Sur l'architecture, j'ai choisi un Strategy Pattern pour pouvoir comparer deux méthodes (une avec gradient, une sans) derrière une interface commune, sans dupliquer la logique d'orchestration entre l'entraînement et la future API d'inférence.

---

# Journal de bord - Smart Inspection Edge Platform

## Date : session SIP#1

### 1. Contexte

Écriture des trois fichiers de configuration YAML (common.yaml, padim.yaml, stfpm.yaml) et du module config/loader.py chargé de les localiser, charger et fusionner. Objectif secondaire : comprendre en profondeur les mécanismes CNN (convolution, feature maps,..) nécessaires pour justifier les choix de configuration de PaDiM/STFPM.

### 2. Ce qui a été fait concrètement

3 fichiers YAML finalisés dans configs/ : common.yaml (backbone, layers, seed, cudnn_deterministic, device, input_size, chemins data/paths), padim.yaml (configuration spécifique à PaDiM), stfpm.yaml (configuration spécifique à STFPM).

config/loader.py complet, 5 fonctions :

get_root_dir() : ancrage repo via remontée des parents jusqu'à pyproject.toml

\_get_config_path() (privée) : résout root_dir/configs/<fichier>

read_yaml() : charge un YAML avec gestion d'erreurs (FileNotFoundError, yaml.YAMLError)

resolve_config_paths() : charge common.yaml et résout les sections data/paths en objets Path absolus, laisse params intact

merge_yaml() : fusionne config commune + config spécifique, priorité au spécifique

Script scripts/00_debug_loader.py validant l'ensemble de bout en bout.

Squelette vide de data/dataset.py (AnomalyDataset avec 4 méthodes en pass/TODO).

Lint/format appliqués : ruff check et ruff format passent sans erreur sur loader.py.

### 3. Difficultés rencontrées et comment elles ont été résolues

#### Duplication de résolution de chemin dans read_yaml/get_paths

**Problème :** get_paths appelait get_config_path puis passait un chemin déjà résolu à read_yaml, qui elle-même rappelait get_config_path en interne -

**Résolution :** identification que read_yaml gère déjà seule toute la résolution, donc get_paths/resolve_config_paths doit lui passer le nom de fichier brut, sans jamais appeler get_config_path elle-même.

#### Incompréhension du mécanisme feature maps / canaux / champ récepteur (CNN)

**Problème :** Difficulté centrale de la session : confusion initiale entre "canal" et "filtre", entre l'évolution de la résolution spatiale (diminue en profondeur) et celle du champ récepteur (augmente en profondeur) - perçus à tort comme contradictoires.

**Résolution :** obtenue après plusieurs itérations d'illustrations (calcul de convolution avec vrais chiffres, comparaison visuelle layer1 vs layer3 sur une même rayure) : compris que layer1 (haute résolution, champ récepteur étroit) détecte bien les petits défauts locaux tandis que layer3 (basse résolution, champ récepteur large) les "noie" statistiquement, d'où la nécessité de combiner plusieurs couches pour PaDiM/STFPM plutôt que d'en choisir une seule.

### 4. Décisions d'architecture / conception, avec le raisonnement

n_features absent de stfpm.yaml : PaDiM a besoin de réduire la dimensionnalité avant de calculer une matrice de covariance (coût de calcul/stockage) ; STFPM ne calcule jamais de covariance, sa loss est une distance directe entre features, donc ce paramètre n'a pas de sens pour cette méthode.

freeze_backbone exclu du YAML, fixé en dur dans le code : le gel du teacher dans STFPM n'est pas un hyperparamètre ajustable mais une contrainte structurelle de la méthode.

Masque de zéros plutôt que None pour les images sans défaut : simplifie le code d'évaluation : consommateur (mais négligeable) et évite des vérifications conditionnelles répétitives

Split train/val sera géré par l'orchestrateur d'entraînement, pas par AnomalyDataset : le Dataset doit rester agnostique de la méthode consommatrice - c'est un composant de plus haut niveau qui compose les objets, pas au Dataset de connaître PaDiM ou STFPM.

Fonction _get_config_path marquée privée (préfixe _) : détail d'implémentation interne à loader.py.

Priorité de merge : config spécifique > config commune en cas de conflit de clé, implémentée via l'ordre d'écriture dans {**common, **model}. Le plus spécifique prioritaire sur le plus général.

### 5. Ce que j'ai appris

Sur les CNN, j'ai compris que "plus de canaux" et "moins de résolution spatiale" ne sont pas deux effets contradictoires mais deux mesures indépendantes qui évoluent chacune dans un sens différent, c'est un réel compromis

---

# Journal de bord - Smart Inspection Edge Platform

## Date : début Aout (session CI + AnomalyDataset, en deux temps avec une pause entre les deux)

### 1. Contexte / objectif du jour

Mise en place de la CI GitHub Actions minimale (lint only), puis implémentation complète de AnomalyDataset : (images + masques de segmentation), méthode par méthode avec test isolé à chaque étape.

### 2. Ce qui a été fait concrètement

CI GitHub Actions (.github/workflows/ci.yml) :

Triggers push + pull_request sur main

Steps : actions/checkout@v4 -> actions/setup-python@v4 (3.12) -> pip install -e .[dev] -> ruff check . + ruff format --check .

Pas de job pytest (tests/ vide)

Pas de branch protection (décision explicite, solo workflow)

AnomalyDataset (src/smart_inspection/data/dataset.py), classe complète :

\***\*init\*\***(category, split) : résolution des chemins via resolve_config_paths("common.yaml"), validation d'existence category_path puis split_path, garde-fou ValueError si dataset vide, construction de deux pipelines Compose distincts (transform_image avec Normalize ImageNet, transform_mask sans)

\_discover_samples(split_path) : scan pathlib.iterdir() + is_dir(), label 0 (good) / 1 (defect), construction du chemin masque par correspondance de nom (stem + "\_mask" + suffix) sous ground_truth/<sous-catégorie>/

\***\*len\*\*** : trivial

\***\*getitem\*\***(idx) : chargement PIL à la demande, transform image (Resize/ToTensor/Normalize ImageNet) et masque (Resize/ToTensor seul), tensor de zéros (1, H, W) si pas de masque réel

Tests empiriques : comptage validé par glob manuel indépendant (83/83 sur bottle/test), shapes confirmées (3,256,256)/(1,256,256), masque de zéros vérifié, plage de valeurs image cohérente avec normalisation ImageNet.

### 3. Difficultés rencontrées et comment elles ont été résolues

#### Confusion CI : environnement local vs runner GitHub

**Problème :** je pensais que la machine CI avait "tout ce qu'il faut" comme ma machine locale, donc je ne comprenais pas pourquoi il fallait un step de récupération du code.

**Fausse hypothèse :** assimiler le runner GitHub Actions à mon PC local, alors que c'est une VM vierge et éphémère, détruite après chaque run.

**Résolution :** une fois la distinction posée clairement (VM neuve à chaque fois, rien n'y persiste), j'ai reconstruit moi-même la nécessité d'un step actions/checkout avant tout le reste.

#### Confusion pathlib : navigation entre dossiers frères (test/ -> ground_truth/)

**Problème :** je n'arrivais pas à comprendre comment construire dynamiquement le chemin vers ground_truth/<sous-catégorie>/ à partir d'un sous-dossier de test/, en particulier je pensais que p.name allait "se recalculer" en fonction d'où on en était dans la construction du chemin final.

**Fausse hypothèse :** croire que p.name était une valeur dynamique réévaluée à chaque ligne, plutôt qu'une valeur figée dès l'entrée dans le tour de boucle.

**Résolution :** traçage pas à pas avec des vraies valeurs (p = .../bottle/test/broken_large/, donc p.name = "broken_large", fixe pour tout le tour) - j'ai compris que je récupère juste une string (le nom) à un endroit, que je réutilise ensuite dans une construction de chemin totalement indépendante (split_path.parent / "ground_truth" / p.name / mask_filename). Le mécanisme ne marche que parce que MVTec garantit les mêmes noms de sous-dossiers entre test/ et ground_truth/ - sinon ça pointerait vers un chemin inexistant sans erreur immédiate (construire un Path ne vérifie jamais son existence réelle).

#### Normalisation recalculée sur MVTec au lieu des stats ImageNet fixes

**Problème :** j'ai voulu calculer mean/std sur mes propres images pour la normalisation.

**Fausse hypothèse :** j'avais oublié que le backbone ResNet est pré-entraîné sur ImageNet, donc attend des données normalisées avec les stats ImageNet, pas des stats recalculées sur mon dataset.

**Résolution :** rappel du principe (le backbone a appris avec une distribution de pixels précise, changer cette distribution en entrée casse le bénéfice du pré-entraînement) - utilisation des constantes IMAGENET_MEAN/STD.

### 4. Décisions d'architecture / conception, avec le raisonnement

CI vibe-coder plutôt qu'à la main : ROI pédagogique trop faible sur ce type de tâche.

Validation d'existence category_path avant split_path dans \***\*init\*\*** : ordre choisi selon la hiérarchie réelle des chemins.

Masque de zéros plutôt que None pour les images normales : décision actée en amont, implémentée ici - simplifie le code (pas de branchement conditionnel à chaque usage).

Deux pipelines Compose séparés (transform_image avec Normalize, transform_mask sans) plutôt qu'un seul pipeline partagé : la normalisation ImageNet n'a de sens que pour des valeurs de couleur RGB, pas pour une carte binaire qui sert à indiquer les zones anormales.

Boucle for explicite plutôt que list comprehension pour \_discover_samples : préférée pour la lisibilité et la facilité de debug étape par étape, vu la complexité de la logique imbriquée (label + condition + calcul de chemin masque).

### 5. Ce que j'ai appris

J'ai compris très concrètement pourquoi une CI tourne sur une machine complètement différente de la mienne - une VM neuve à chaque run, sans aucun état persistant - et pourquoi ça impose une étape explicite de récupération du code avant de pouvoir faire quoi que ce soit dessus.

Sur la normalisation ImageNet, j'ai bien intégré que ce n'est pas une convention arbitraire : le backbone pré-entraîné a appris sur une distribution de pixels précise, et si je recalcule mes propres stats ou que je m'arrête à ToTensor() sans Normalize, je casse silencieusement l'hypothèse sur laquelle repose tout le transfer learning - sans qu'aucune erreur ne se déclenche pour me le signaler.

---

# Journal de bord - Smart Inspection Edge Platform

## Date : sessions milieu août 2026

### 1. Contexte

Conception de l'interface commune `AnomalyMethod` avec le Strategy Pattern et de `factory.py`, puis implémentation complète de PaDiM (`__init__`, `fit()`, `predict()`), de la configuration jusqu'au calcul final du score d'anomalie.

Objectif secondaire : valider empiriquement chaque étape du pipeline PaDiM sur la catégorie `bottle` de MVTec AD, plutôt que de considérer l'implémentation comme correcte uniquement parce qu'elle s'exécute sans erreur.

### 2. Ce qui a été fait concrètement

`models/base.py` :

- Classe abstraite `AnomalyMethod(ABC)`.
- Définition de l'interface commune avec deux méthodes abstraites :
  - `fit(train_loader: DataLoader) -> None`
  - `predict(image: Tensor) -> tuple[float, Tensor]`

`models/factory.py` :

- Dictionnaire statique `METHODS` associant les noms de méthodes à leurs classes :
  - `"padim"` -> `PaDiM`
  - `"stfpm"` -> `STFPM`
- Fonction `create_method(method_name) -> AnomalyMethod`.
- Gestion explicite d'une méthode inconnue avec `ValueError`.

`models/padim/model.py` - implémentation complète de PaDiM :

- `__init__` :
  - chargement de `common.yaml` et `padim.yaml` ;
  - merge de la configuration commune et spécifique ;
  - fixation de la reproductibilité avec `torch.manual_seed` et `cudnn.deterministic` ;
  - chargement du ResNet18 pré-entraîné ;
  - gel de l'ensemble des paramètres du backbone ;
  - passage en mode `.eval()` permanent ;
  - résolution dynamique du device ;
  - installation de forward hooks sur `layer1`, `layer2` et `layer3` ;
  - utilisation d'un pattern `make_hook(layer_name)` avec closure pour conserver le nom de chaque couche.

- `fit()` :
  - extraction des features par batch sous `torch.no_grad()` ;
  - récupération des sorties des trois couches via les hooks ;
  - upsampling de `layer2` et `layer3` vers la résolution spatiale de `layer1` avec `F.interpolate` ;
  - concaténation des trois couches sur la dimension des canaux ;
  - restructuration des features avec `permute` puis `reshape` afin d'obtenir une représentation `(H*W, N, C)` ;
  - réduction dimensionnelle aléatoire et reproductible de 448 à 100 canaux avec `torch.randperm` ;
  - conservation des mêmes indices pour la phase de prédiction ;
  - calcul vectorisé de la moyenne pour chaque position spatiale ;
  - calcul vectorisé de la covariance par position ;
  - utilisation de la correction de Bessel avec division par `N-1` ;
  - régularisation des matrices de covariance avec `+ epsilon * I`.

- `predict()` :
  - extraction des features d'une seule image avec exactement le même pipeline que pendant `fit()` ;
  - upsampling et concaténation des trois couches ;
  - application des mêmes indices de réduction dimensionnelle utilisés pendant l'entraînement ;
  - calcul vectorisé de la distance de Mahalanobis pour chaque position spatiale ;
  - calcul de la carte d'anomalie ;
  - score global obtenu avec le maximum de la carte ;
  - retour du score et de la carte d'anomalie.

Scripts de debug :

- `02_debug_padim.py` utilisé pour vérifier séparément :
  - le device utilisé ;
  - le gel des paramètres du backbone ;
  - le mode `eval()` ;
  - le fonctionnement des hooks ;
  - les shapes obtenues pendant `fit()` ;
  - puis le comportement discriminant complet entre une image `defect` et une image `good`.

Validation empirique :

- test réalisé sur la catégorie `bottle` de MVTec AD ;
- vérification du pipeline jusqu'au score final ;
- comparaison d'une image `defect` avec une image `good` pour vérifier que le modèle produit bien un signal d'anomalie cohérent.

### 3. Difficultés rencontrées et comment elles ont été résolues

#### Sur-ingénierie de `factory.py` avant l'existence des méthodes concrètes

**Problème :** je voulais déjà construire la factory avec des imports vers `PaDiM` et `STFPM`, alors que les classes n'étaient pas encore réellement implémentées.

**Cause :** volonté d'avancer simultanément sur plusieurs blocs au lieu de suivre une progression où chaque composant est testable avant de construire le suivant.

**Résolution :** reconnaissance qu'un fichier dépendant de classes inexistantes ne peut pas être validé correctement. La factory a donc été finalisée après avoir disposé d'au moins une implémentation concrète.

#### Pic mémoire critique pendant `fit()`

**Problème :** le premier lancement de `fit()` a fait chuter la RAM disponible jusqu'à environ 187 Mio sur une machine disposant de 15 Go, provoquant le crash de VSCode. Aucun swap n'était configuré.

**Première intuition erronée :** je pensais principalement au volume des features concaténées à 448 canaux.

**Diagnostic :** la réduction `n_features=100` arrivait effectivement après la construction complète du tensor à 448 canaux. Elle ne supprimait donc pas le pic mémoire principal.

La cause déterminante était surtout l'absence de `torch.no_grad()` autour des forwards : PyTorch construisait inutilement un graphe de calcul alors qu'aucun gradient n'est utilisé dans PaDiM.

**Résolution :** ajout de `torch.no_grad()` autour de la boucle d'extraction des features.

**Validation :** surveillance simultanée de la mémoire avec `free -h -s 1`, permettant de constater empiriquement un pic nettement plus faible, avec environ 6,8 Gio disponibles après correction.

#### Distance de Mahalanobis : ordre des multiplications matricielles

**Problème :** premiers essais avec des variables `centered_line` et `centered_column` dont les noms ne correspondaient pas réellement aux shapes, entraînant une confusion sur l'ordre des `torch.matmul`.

**Cause :** difficulté à visualiser les multiplications matricielles batchées avec des vecteurs considérés successivement comme lignes et colonnes.

**Résolution :** retour à un exemple manuel avec deux canaux et calcul complet jusqu'à un scalaire égal à `10`.

Cela a permis de reconstruire la formule :

`(x - μ)ᵀ Σ⁻¹ (x - μ)`

avant de la traduire en opérations PyTorch vectorisées.

#### Confusion `torch.Size` vs `Tensor`

**Problème :** tentative d'appeler `.shape` sur un objet obtenu par `.shape[2:]`.

**Cause :** supposition que tout objet manipulé dans le pipeline était nécessairement un tensor.

**Résolution :** clarification de la distinction :

- un `Tensor` contient les données ;
- `.shape` renvoie une description de ses dimensions ;
- `.shape[2:]` renvoie un objet `torch.Size`, qui décrit déjà des dimensions et n'est pas lui-même un tensor.

### 4. Décisions d'architecture / conception, avec le raisonnement

`predict()` prend une seule image et jamais un `DataLoader` : le score d'une image de test est indépendant des autres images. À l'inverse, `fit()` doit disposer de l'ensemble des images normales d'entraînement pour estimer correctement les moyennes et covariances. Cette séparation correspond également au futur endpoint FastAPI `/predict`, qui recevra une image par requête.

`factory.py` reste une fonction simple plutôt qu'une classe : aucune donnée d'état ne doit être conservée entre deux appels.

Dictionnaire `METHODS` plutôt qu'une chaîne de `if/elif` : l'ajout d'une future méthode, (Ex : PatchCore), nécessite uniquement l'ajout d'une entrée dans le dictionnaire au lieu de modifier une logique conditionnelle existante.

Score global obtenu avec `max()` plutôt qu'avec la moyenne de la carte d'anomalie : un défaut industriel peut être extrêmement localisé. Une éraflure sur quelques pixels serait diluée par une moyenne sur l'ensemble des 4096 positions spatiales, alors que le maximum conserve directement le signal de la position la plus anormale.

Réduction des canaux uniquement après concaténation des trois couches : réduire séparément `layer1`, `layer2` et `layer3` imposerait artificiellement une répartition des 100 features entre les couches. La sélection aléatoire appliquée après concaténation permet au contraire de sélectionner librement parmi les 448 canaux issus des trois niveaux du backbone.

`+ epsilon * I` conservé après la division par `N-1` : choix volontaire de respecter l'ordre de la formule méthodologique de PaDiM plutôt que de déplacer arbitrairement la régularisation dans le calcul de covariance.

Correction de Bessel avec `N-1` systématiquement : reste celui correspondant à la convention statistique standard pour un échantillon.

### 5. Ce que j'ai appris

J'ai surtout compris que la mémoire PyTorch ne dépend pas uniquement de la taille des tensors que je vois explicitement dans mon code. Le framework peut construire implicitement un graphe de calcul complet pendant les forwards. Dans un algorithme comme PaDiM, où aucun gradient n'est nécessaire, `torch.no_grad()` est donc une condition importante de bon fonctionnement et pas une simple optimisation. Dans mon cas, son ajout a transformé un pipeline qui faisait crasher l'IDE en un pipeline exploitable.

Pour les calculs matriciels comme la covariance ou la distance de Mahalanobis, j'ai adopté une nouvelle méthode de travail : avant d'écrire une vectorisation PyTorch complexe, je vérifie les shapes et l'ordre des opérations avec un petit exemple numérique calculé à la main. Cela permet de comprendre réellement ce que fait la vectorisation au lieu de simplement reproduire une formule ou du code trouvé ailleurs.

---

# Journal de bord - Smart Inspection Edge Platform

## Date : fin août 2026 (plusieurs sessions)

### 1. Contexte

Finalisation du chantier `factory.py`/PaDiM entamé la veille (`03_debug_factory.py`, refactor de PaDiM en fonctions pures testables, tests mathématiques Bessel/régularisation/Mahalanobis), puis implémentation complète de STFPM depuis zéro : compréhension du papier, `__init__` (teacher/student), `fit()` (vraie boucle de gradient), `predict()`.

### 2. Ce qui a été fait concrètement

**Factory / PaDiM - clôture du chantier de la veille :**

- `scripts/debug/03_debug_factory.py` : script d'intégration `create_method("padim") -> fit() -> predict()`, avec recherche d'une image bad/good par label (pas d'index en dur), assertion explicite `score_bad > score_good`.
- Refactor de PaDiM : extraction de `_compute_covariance`, `_regularize_covariance`, `_compute_mahalanobis_distance` en `@staticmethod` pures (aucune dépendance à `self`), permettant de les tester sans charger le ResNet18.
- `tests/test_padim.py` : 3 tests mathématiques avec valeurs calculées à la main - correction de Bessel (vecteur `[1,2,3]`, variance attendue = 1), régularisation `+εI` avec inverse d'une matrice 2×2 calculé à la main, Mahalanobis avec Σ=identité (réduction à une distance euclidienne au carré).
- Vérification croisée CPU (machine1) vs GPU (machine2) : ratio bad/good passe de ~8x (CUDA) à ~2.1x (CPU) sur le même code - confirmé comme un effet de non-reproductibilité de `torch.randperm` entre devices, pas une régression.

**STFPM - implémentation complète :**

- Lecture du papier STFPM (arXiv:2103.04257), sections 3.2/3.3, reconstruction de la compréhension conceptuelle (teacher gelé, student appris uniquement sur images normales).
- `__init__` : deux ResNet18 séparés (teacher `weights="DEFAULT"`, student `weights=None`), `requires_grad` opposé, hooks doubles via `make_hook(features_dict, layer_name)` généralisée pour écrire dans deux dictionnaires séparés (`self.teacher_features`/`self.student_features`) sans collision.
- 3 méthodes mathématiques `@staticmethod` testées isolément : `_normalize_features` (wrapper `F.normalize`), `_compute_distillation_loss` (Eq.1, `0.5 * sum((F_t-F_s)²)`), `_compute_spatial_mean_loss` (Eq.2, `mean(dim=(2,3))`).
- `fit()` : ABC `AnomalyMethod.fit()` modifiée pour accepter `val_loader: DataLoader | None = None` (rétrocompatible avec PaDiM). Boucle epoch complète (train avec `backward()`/`step()`, eval sous `no_grad()`), checkpoint sur meilleure val loss via `copy.deepcopy(state_dict())`, guard rails (`ValueError` si `val_loader` manquant ou aucun best model trouvé).
- `predict()` : forward teacher/student sur une image, upsampling de chaque anomaly map par couche vers la taille de l'image originale (pas vers `layer1` comme PaDiM), combinaison par produit élément-wise, score = `max()`.
- `tests/test_stfpm.py` : 6 tests (normalize x3, distillation loss x2, spatial mean x1).
- Validation empirique complète sur `bottle`, 100 epochs réelles (hyperparams du papier) : train loss 1.66 -> 0.088, val loss 1.95 -> 0.094, score final bad=0.0858 vs good=0.0035 (ratio ~24x).

### 3. Difficultés rencontrées et comment elles ont été résolues

#### Vectorisation vs boucle Python - pourquoi PyTorch n'a pas besoin de boucle explicite

**Problème :** je pensais qu'il fallait boucler explicitement sur chaque position spatiale `(i,j)` pour appliquer la normalisation, comme le suggérait la notation indicielle du papier.

**Résolution :** Compréhension de ce que fait PyTorch "sous le capot" - une boucle existe bien, mais écrite en C++/CUDA compilée, exécutée en parallèle massif sur GPU, pas en Python interprété.

#### `state_dict()` par référence vs `copy.deepcopy`

**Problème :** je ne savais pas comment faire un snapshot des poids du student à un epoch donné pour les restaurer plus tard.

**Résolution :** compris que state_dict() seul retourne des références vers les tensors vivants du modèle, pas un snapshot indépendant - utilisation de copy.deepcopy(state_dict()) pour obtenir une vraie copie figée.

### 4. Décisions d'architecture / conception, avec le raisonnement

Signature de l'ABC `AnomalyMethod.fit()` étendue avec `val_loader: DataLoader | None = None` plutôt que deux méthodes distinctes ou un `fit()` qui fait le split lui-même : le split 80/20 est une responsabilité de construction des données (futur `training/`), pas de la méthode ML elle-même - PaDiM n'en a jamais eu besoin, STFPM en a besoin structurellement à cause du gradient. Python ne vérifie que le nom des méthodes abstraites, pas leur signature complète, donc PaDiM reste valide sans modification.

Deux dictionnaires séparés (`self.teacher_features`/`self.student_features`) plutôt qu'un seul dictionnaire avec des clés préfixées : (`self.teacher_features["layer1"]` se lit sans ambiguïté) et évite un risque de collision d'écriture entre les deux forwards.

Optimizer recréé à chaque appel de `fit()` plutôt que stocké dans `__init__` : repartir sans état de momentum résiduel d'un éventuel entraînement précédent sur la même instance.

Le score scalaire final ne sert qu'à la classification image-level, la localisation reste disponible dans la carte complète avant `.max()`.

Produit element-wise et somme pondérée entre couches (Eq. 3) volontairement **non extraits** en méthodes testables séparées : logique triviale (3 lignes, pas de risque mathématique caché), même raisonnement déjà appliqué à `mean()` seul sur PaDiM

### 5. Ce que j'ai appris

Sur le plan technique, j'ai consolidé ma compréhension de la distillation de connaissance (knowledge distillation) au-delà du cas de STFPM : un student qui n'apprend que sur des données normales n'a statistiquement aucune raison de bien généraliser sur des patterns jamais vus, ce qui est le principe fondateur de toute la détection d'anomalie par distillation - mais j'ai aussi identifié moi-même une limite réelle à cette hypothèse (un student qui généraliserait "trop bien" pourrait produire un faux négatif), ce qui m'a permis de comprendre que c'est un pari empirique validé par les résultats du papier, pas une garantie mathématique absolue.

---

# Journal de bord - Smart Inspection Edge Platform

## Date : début septembre 2026 (training/ + evaluation/)

### Contexte

`training/train.py` (orchestrateur générique) + `evaluation/evaluate.py` (AUROC sur test set complet). Remplace le ratio bad/good à 2 échantillons par une métrique statistiquement valide, comparable entre PaDiM et STFPM.

### Réalisations

- Property abstraite `validation_split_ratio` (`float | None`) sur `AnomalyMethod` : PaDiM --> `None`, STFPM --> son ratio config. Permet à `train()` de rester générique.
- `train()` : split conditionnel (`random_split`), `drop_last=False` forcé sur val_loader, guard clauses (ratio hors (0,1), split vide, loader vide), un seul `Generator(seed)` partagé entre split et shuffle du train_loader (y compris sans split, cas PaDiM) --> reproductibilité complète.
- `batch_size` sorti des yaml spécifiques --> `common.yaml` seul, argument obligatoire de `train()` : paramètre d'exécution générique, pas structurel (contrairement à `validation_split_ratio`) --> pas ajouté à `AnomalyMethod`.
- `evaluate()` : dé-batch image par image pour `predict()`, guard clauses (loader vide / classe unique)
- Validé empiriquement : `06_debug_training.py` (86% poids student changés, état interne PaDiM non nul), `07_debug_evaluate.py` (AUROC STFPM 0.996 / PaDiM 0.993, 10 epochs, run de validation pipeline - pas le run définitif).

### Difficultés

- J'avais supprimé par erreur la garde "classe unique" dans `evaluate.py`.
- **AUROC non proche de 0.5** : le premier signale une inversion de convention (label/score), le second une vraie absence de discrimination. `1 - 0.15 = 0.85` : diagnostic à corriger, pas modèle à réentraîner.

### Décisions clés

- `validation_split_ratio` en property abstraite plutôt que `isinstance`/YAML rouvert dans `train()`
- `batch_size` diffère de `validation_split_ratio` : exécution générique vs structurel méthode --> deux traitements différents.

### Appris

AUROC bas = deux diagnostics distincts selon qu'il est proche de 0 ou de 0.5. Arbitrage contrat abstrait vs paramètre d'orchestrateur : dépend de la nature structurelle ou non du paramètre.

_Suite : script de comparaison PaDiM/STFPM (run définitif 100 epochs) + benchmark FPS/RAM. Dette vus aussi : boilerplate config dupliqué (les yaml) (scripts debug : 03/05/06/07), couplage evaluate.py aux labels {0,1}._

---

# Journal de bord - Smart Inspection Edge Platform

## Date : fin septembre 2026 (audit complet + correction critique PaDiM)

### Contexte
Finalisation du benchmark FPS/VRAM entre PaDiM et STFPM, config des scripts debug refactorée et correction critique de PaDiM.
Audit externe en lecture seule sur l'ensemble du repo par Claude (Opus) (code, tests, CI, historique git). Révèle un bug de fond dans la comparaison PaDiM/STFPM qui invalide les résultats FPS/VRAM produits jusqu'ici.

### Réalisations
- **Bug critique corrigé** : `predict()` de PaDiM recalculait `torch.linalg.inv(cov)` (4096 inversions de matrices 100×100) à chaque appel, alors que `cov` est fixe après `fit()`. Précalcul de `self.inv_cov` une seule fois, dans `fit()` et `load()`.
- Tri déterministe du dataset (`sorted()` sur `iterdir()`/`glob()`) : l'ordre des fichiers dépendait du système de fichiers, cassant la reproductibilité du split seedé entre machines (c'est pour cela que j'avais des léger écarts entre mes deux laptops).
- `.convert("RGB")` à l'ouverture des images : les catégories MVTec en niveaux de gris (grid, screw, zipper) plantaient sur `Normalize` (1 canal vs 3 attendus) - jamais détecté faute de les avoir testées.
- Masques redimensionnés en interpolation NEAREST (pas bilinéaire) : préserve des valeurs binaires 0/1, nécessaire pour un futur AUROC pixel.
- Résultats de référence régénérés sur deux catégories (bottle, cable) après correction.

### Résultats : avant/après correction (cable)
| | avant (bug) | après (corrigé) |
|---|---|---|
| PaDiM FPS | 42 (23.7ms) | 676 (1.5ms) |
| PaDiM VRAM/inférence | 488 Mo | 13 Mo |
| Conclusion | STFPM ~11× plus rapide | **PaDiM plus rapide que STFPM** |

Le classement s'inverse complètement : PaDiM (un seul forward + une multiplication) est plus rapide que STFPM (deux forwards), au prix d'une mémoire résidente plus élevée (mean/cov/inv_cov gardés en mémoire, ~375 Mo).

### Difficultés
- **Le bug a survécu à plusieurs relectures ciblées** de `predict()` sans être repéré - seul un audit complet (Claude) au bout d'un certain moment l'a fait ressortir. Principe retenu : review par IA à intégrer plus régulièrement.
 
### Décisions clés
- Adoption d'une approche plus systématique pour les benchmarks : séparation claire entre préparation des données, chargement des modèles et mesure des performances (FPS/VRAM).
- Nettoyage de la mémoire GPU entre les benchmarks pour éviter les interférences et obtenir des mesures plus fiables.

### Appris
Un bug de performance peut invalider une conclusion scientifique entière sans jamais produire d'erreur ni de résultat visiblement faux - PaDiM "fonctionnait", donnait le bon score, juste beaucoup trop lentement pour de mauvaises raisons (ma faute, mauvaise réflexion scientifique lors du développement). Mesurer une différence entre deux implémentations ne prouve rien sur les algorithmes eux-mêmes tant que chaque implémentation n'a pas été auditée pour des inefficacités évitables.

*Suite : review Claude Code globale sur l'état corrigé. Reste ouvert : AUROC pixel + visualisation des cartes d'anomalie*

---

## Date : début octobre 2026 (localisation, fidélité PaDiM, visualisation)

### Contexte
Ajout de la localisation (AUROC pixel) et de figures qualitatives. Relecture de `padim/model.py` contre le papier source (arXiv:2011.08785, section IV-B) : des écarts existaient depuis l'implémentation initiale.

### Réalisations
- **PaDiM aligné sur le papier** : racine carrée de l'Éq. 2 (le code calculait le carré de la distance), interpolation bicubic, flou gaussien σ=4 (`sigma` dans `padim.yaml`), `param_input_size` manquant ajouté. La carte est upsamplée à 256×256 dans `predict()`, comme celle de STFPM, pour être comparée au masque.
- **`evaluate()`** : AUROC pixel en plus de l'AUROC image. Accumulation en tableaux numpy (`np.concatenate`) plutôt qu'en listes Python, gain mémoire. Gardes : carte et masque de même taille, deux classes présentes côté image et côté pixel.
- **Dataset** : ordre de scan trié, `.convert("RGB")`, masques redimensionnés en NEAREST pour rester binaires.
- **Checkpoints** : `save()`/`load()` sur `AnomalyMethod`, `save_path` optionnel dans `train()`. PaDiM stocke `cov` et recalcule `inv_cov` au chargement.
- **`evaluation/benchmark.py`** : FPS (médiane, p95, p99) et VRAM (résidente, pic, coût par inférence), avec warm-up et `synchronize()`.
- **Visualisation** : `visualization/plots.py`, `io/save_fig.py` et `scripts/generate_anomaly_figures.py`. Figures par méthode et 4 figures de comparaison, avec une échelle de couleur commune par méthode.
- **Figures de référence committées** : `comparison/` (4) et `high_normal/` (6) du run `figure_20261004_123157`. Les 12 autres restent en local, les comparaisons les recouvrent.
- **Tests** : 89 passent en CI (`-m "not integration"`), 6 exclus car ils exigent le dataset ou les poids.

### Résultats (après corrections, GPU)
| | bottle image | bottle pixel | cable image | cable pixel |
|---|---|---|---|---|
| STFPM | 1.000 | 0.988 | 0.914 | 0.947 |
| PaDiM | 0.994 | 0.983 | 0.852 | **0.974** |

| FPS (cable) | PaDiM | STFPM |
|---|---|---|
| avant correction `inv_cov` | 42 | 463 |
| après toutes les corrections | 582 | 456 |

- VRAM résidente : PaDiM ~375 Mo (mean, cov, inv_cov), STFPM ~102 Mo. Coût par inférence : ~13 Mo contre ~10 Mo.
- Sur cable, le classement s'inverse entre image et pixel : STFPM gagne en image, PaDiM en pixel.

### Ce que montrent les figures (cable)
- Sur les anomalies bien détectées, les deux cartes tombent sur le masque. STFPM est plus net, PaDiM plus étalé (σ=4, grille 64×64).
- Les anomalies les plus ratées par les deux méthodes ont des masques en anneau autour d'un câble.
- Chaque méthode a aussi ses ratés propres : STFPM ne voit pas un grand défaut d'isolant que PaDiM détecte très bien, PaDiM localise un minuscule défaut que STFPM manque.
- L'échelle commune écrase la colonne STFPM en bleu (sa carte est un produit de trois cartes) : bleu veut dire score bas par rapport au pire du test set, pas absence de structure.

### Difficultés
- **Optimisation mémoire numpy** : en passant `pixel_labels` d'une liste de scalaires à une liste de tableaux, `.count()` devenait faux et `set()` plantait. passé sur les fonctions propre à numpy
- **Mismatch de résolution** (64×64 contre 256×256) : `F.interpolate` attend `(N,C,H,W)`, et un `reshape` dans le mauvais ordre ne lève aucune erreur.
- **`.append()` contre `.extend()`** sur les listes de valeurs pixel.

### Décisions clés
- Upsampling dans `predict()` plutôt que dans `evaluate()` : l'évaluation reste générique, chaque méthode garantit un contrat de sortie cohérent.
- Cartes gardées en mémoire (~39 Mo par méthode sur cable) plutôt qu'un second passage de `predict()`.
- Comparaison : une liste d'`idx` par figure, choisie par une méthode, puis les cartes des deux méthodes lues aux mêmes `idx`.
- Un run de référence par type de résultat (JSON et figures), au lieu de committer chaque lancement.
- Pas de module séparé pour valider les métadonnées de checkpoint : deux classes concrètes ne le justifient pas. 

### Appris
- Une explication (faux positifs ponctuels de PaDiM) ne vaut que si les figures la confirment. Elles montrent surtout des anomalies à score bas, communes aux deux méthodes.
- Deux métriques voisines (AUROC image et pixel) peuvent classer deux méthodes à l'envers sans contradiction.
- Vérifier une implémentation contre le papier, y compris les détails expérimentaux (interpolation, filtre, racine carrée).

*Suite : AUROC par type de défaut, échelle de couleur de STFPM, README avec tableau de résultats, config injectée dans les constructeurs.*

---

## Date : 10 octobre 2026 (AUROC par type de défaut sur cable)

### Contexte
Hypothèse du 4 octobre, non vérifiée : sur cable, les anomalies les plus ratées par PaDiM et STFPM seraient des `cable_swap` (anomalie logique). Les AUROC globaux (image : STFPM 0.914, PaDiM 0.852) mélangent 8 types de défauts et ne peuvent ni confirmer ni infirmer cette hypothèse. **Objectif :** un AUROC image par type de défaut, avec les checkpoints du tableau de référence (pas de réentraînement).

### Prédictions (écrites et commitées AVANT exécution sur cable)
- Classement du plus facile au plus dur (intuition, sans justification formelle, un seul classement pour les deux méthodes) : `combined`, `bent_wire`, `cut_inner_insulation`, `cut_outer_insulation`, `missing_cable`, `missing_wire`, `cable_swap`, `poke_insulation`.
- Règle de décision sur `cable_swap` :
  - AUROC <= 0.65 : compatible avec « aveugle » ;
  - entre 0.65 et 0.8 : non concluant
  - si >= 0.8 : hypothèse réfutée.

### Réalisations
- **Dataset** : chaque échantillon porte une clé `subcategory` = nom du dossier sous le split (`"good"` pour les normales), servie par `__getitem__`. Nommée `subcategory` et non `defect_category` pour ne pas la confondre avec `category` (`cable`, `bottle`…), le niveau au-dessus.
- **`evaluate()`** : collecte la sous-catégorie dans la même itération que le score et le label (alignement garanti même avec `shuffle=True`).
- **`_evaluate_by_subcategory`** : fonction pure (scores, labels, sous-catégories). Pour chaque type de défaut, positifs = ses échantillons anormaux ; négatifs = toutes les images normales, les mêmes pour tous les types (`n_normal` constant). Les normales sont repérées par le label et non par le nom `"good"` (métrique indépendante des noms de dossiers MVTec). Sortie triée par nom (JSON reproductible), `zip(strict=True)`, garde-fous « aucune normale » et « aucun défaut ».
- **Tests** : 6 tests sur la fonction pure (longueurs différentes, sans normale, sans défaut, ordre trié, normales identifiées par le label).
- À compléter : tests d'`evaluate()` (clé présente, résultat identique avec et sans `shuffle`), script d'évaluation depuis les checkpoints, résultats.
- Test d'`evaluate()` ajoutés (clé subcategory présente et correcte)
- reference_check (comparaison d'un run avec un rapport de référence et ses tests)
- Script `compare_method_by_subcategory.py` : évalue les checkpoints par type de défaut, vérifie
contre le rapport de référence, puis sauvegarde le rapport avant de lever l'erreur si la vérification échoue.
- Résultats ajoutés au journal

### Difficultés
- **Garde-fous placés dans la boucle de tri** : ils s'exécutaient à chaque échantillon et levaient une erreur dès le premier. Cause : l'indentation. Le tri et la validation sont deux boucles successives, pas imbriquées.

### Décisions clés
- Une seule passe de `predict()` sur le test de cable, puis regroupement par type.
- Image-level d'abord. L'AUROC pixel par type reste à concevoir : le choix des pixels négatifs change le sens du chiffre.
- Pas de réentraînement : chargement des checkpoints du run de référence, pour rester comparable au tableau existant.

### Résultats
Run `compare_method_bysubcategory_cuda_20261011_162757` (checkpoints `*_cable_cuda_20261003_171414`, GPU). Vérification contre le rapport de référence : **pass**, écarts image et pixel de 0.0 pour les deux méthodes (modèles rechargés depuis le `.pt` = modèles évalués en mémoire le 3 octobre).

AUROC image par type de défaut (négatifs : les 58 images normales pour chaque type), lignes dans l'ordre de la prédiction (rang 1 = le mieux détecté, AUROC le plus haut) :

| Rang prédit | Rang réel STFPM | Rang réel PaDiM | Type | n anomalies | STFPM | PaDiM | delta (STFPM - PaDiM) |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 3 | `combined` | 11 | 0.959 | 0.933 | +0.027 |
| 2 | 1 | 1 | `bent_wire` | 13 | 1.000 | 0.963 | +0.037 |
| 3 | 2 | 4 | `cut_inner_insulation` | 14 | 0.995 | 0.863 | +0.132 |
| 4 | 6 | 2 | `cut_outer_insulation` | 10 | 0.888 | 0.950 | −0.062 |
| 5 | 5 | 7 | `missing_cable` | 12 | 0.902 | 0.764 | +0.138 |
| 6 | 4 | 6 | `missing_wire` | 10 | 0.924 | 0.772 | +0.152 |
| 7 | 8 | 8 | `cable_swap` | 12 | 0.736 | 0.716 | +0.020 |
| 8 | 7 | 5 | `poke_insulation` | 10 | 0.879 | 0.855 | +0.024 |
| | | | **Global** | 92 | 0.914 | 0.852 | +0.061 |

**Observations :**
- Les checkpoints rechargés redonnent exactement les AUROC de référence (écart 0.0) mais sur la même machine et le même GPU.
- J'avais défini une hypothèse (cf. `## Date : 10 octobre 2026 (AUROC par type de défaut sur cable)`) :
 0.65 à 0.8 = non concluant. Résultat : STFPM 0.736, PaDiM 0.716, donc **non concluant**.
- A posteriori : `cable_swap` a l'AUROC le plus bas des 8 types pour les deux méthodes.
- PaDiM est seulement meilleur pour `cut_outer_insulation`.
- 10 à 14 images par type de défaut reste peu, ce qui peut limiter la robustesse de mes conclusions.


*Suite : Section du 17 octobre 2026*

---

## Date : 17 octobre 2026 (cable_swap : lecture des cartes et argmax)

### Contexte
Le 11 octobre, l'AUROC par type de défaut est ressorti « non concluant » sur `cable_swap` (STFPM 0.736, PaDiM 0.716).
Question : sur les 12 images `cable_swap`, où tombe la zone la plus chaude des cartes, et est-elle sur le masque ?

Règle de lecture de l'argmax, posée avant l'exécution : anneau blanc seulement, le cuivre n'est pas une anomalie dans cette catégorie.

### Réalisations
- `scripts/generate_cable_swap_figures.py` : charge les checkpoints du 3 octobre (`*_cable_cuda_20261003_171414`), prédit tout le test de cable, puis produit pour chaque méthode :
  - 12 figures en échelle de couleur par image (titre : indice, score, ratio au max global de la méthode) ;
  - un diagnostic des scores (médiane et max des 58 normales, scores des 12 swap triés, nombre de swap au-dessus du max normal, max global) ;
  - un test d'argmax : le pixel max de la carte est-il dans le masque ?
- Rapport : `outputs/reports/results_cable_swap_20261017_144717.json`.
- `generate_plots` : paramètre optionnel `title`.

### Résultats
Diagnostic des scores (cable test, 58 normales, 12 swap) :

| | PaDiM | STFPM |
|---|---|---|
| Médiane des normales | 11.75 | 0.00761 |
| Max des normales | 21.91 | 0.01418 |
| Plage des scores swap | 11.44 à 21.52 | 0.00665 à 0.01558 |
| Swap au-dessus du max normal | 0 | 1 |
| Max global | 28.12 (bent_wire) | 0.0801 (bent_wire) |

Argmax dans le masque (règle : anneau blanc seulement, le cuivre n'est pas une anomalie dans cette catégorie) :
- PaDiM : **5/12** (images 15, 16, 18, 20, 23).
- STFPM : **4/12** (images 15, 18, 21, 24).
- Les deux : 15, 18. Aucun des deux : 13, 14, 17, 19, 22.
- Hors #22 : PaDiM 5/11, STFPM 4/11.

Image 22 : les deux argmax sont dans le coin bas-gauche (PaDiM (255, 11), STFPM (242, 18)). C'est le score swap le plus haut des deux méthodes.
Ratio au max global, STFPM : non interprétable. Les swap vont de 0.083 à 0.195 et la pire normale est à 0.177, dans la même plage.

### Observations
- Image 22 : les deux argmax sont dans le coin bas-gauche. Sur les 3 normales `high_normal` de chaque méthode, ce coin est aussi chaud sur la carte, ce qui suggère un artefact plutôt qu'une anomalie réelle.
- Hors #22, l'argmax tombe sur l'anneau dans 5 images sur 11 (PaDiM) et 4 sur 11 (STFPM), avec du bruit autour des zones chaudes.
- Le score max pose un souci : une image normale à score élevé peut être confondue avec une anomalie. Ici, le max normal de PaDiM (21.91) dépasse les 12 scores swap.

### Hypothèses non vérifiées
- L'AUROC de `cable_swap` est peut-être flatté par #22. Non testé. Il faudrait recalculer l'AUROC en excluant cette image.

### Limites
n = 12 images. Plancher de hasard de l'argmax non mesuré : 5/12 et 4/12 ne se lisent ni comme « localise » ni comme « ne localise pas ».

### Difficultés
- Manipuler les dictionnaires à travers deux boucles imbriquées (méthode, puis image), en gardant la cohérence entre indices et scores.
- Confusion d'indice entre #22 et #23, détectée en recoupant le ratio du titre (0.195 × 0.0801 ≈ 0.0156).

### Décisions clés
- Règle de l'argmax : anneau seulement, le cuivre n'est pas considéré comme une anomalie.
- Chantier clos sans mesurer le plancher de hasard : Le but était de comprendre le comportement des méthodes sur les anomalies visibles, pas de produire une évaluation statistique rigoureuse.
- Les 24 PNG ne sont pas versionnés.

### Appris
- Apprendre à se poser les bonnes questions : qu'est-il judicieux de mesurer et d'analyser qualitativement, et qu'est-ce qui est du bruit à ignorer pour ne pas fausser l'interprétation.
- Certaines métriques peuvent être trompeuses si elles ne sont pas interprétées dans le bon contexte.
- Un score élevé n'est pas une localisation correcte : #22 a le score swap le plus haut des deux méthodes, pour une mauvaise raison.
- Une impression se vérifie par un comptage : « ça a l'air de bien détecter » a été confronté à 5/12 et 4/12.
- Le ratio au max global est difficile à interpréter lorsque la distribution des scores est lourde comme STFPM.

---

## Date : 5 novembre 2026 (seuil de décision : règle pré-enregistrée)

### Règle (écrite avant exécution)

1. **Seuil de décision :** T = P95 des scores de calibration ; une image est anormale si score >= T.
2. **Découpage n images :** J'ai décidé de prendre 224 - floor(224 * 0.2) = 224 - 44 = 180. Donc 44 qui seront utilisées pour la calibration
puis de ces 180 images, 20% seront utilisées pour la validation (STFPM), soit 36 images ce qui ramène à 144 images pour l'entraînement.
Pour assurer une homogénéité dans la comparaison des modèles, 144 seront utilisées pour l'entraînement de chaque modèle. Même découpage et même seed  [Voir le fichier de configuration](../configs/common.yaml).
3. **Percentile P95 :** J'ai décidé d'utiliser un p95 des scores de chaque modèle entraîné calculé sur les 44 normales de calibration comme seuil de décision pour déterminer si une image est considérée comme anormale.
np.percentile(scores, 95) avec une interpolation linéaire. Un seuil trop haut risque de classer trop peu d'images comme anormales (faux négatifs), tandis qu'un seuil trop bas risque de générer trop de faux positifs (déclaré anomalie à tort).
4. **Convention sur le signe :** J'utilise un >= et non > car pour moi une image avec un score exactement égal au seuil doit être considérée comme anormale (meme si cela risque d'arriver rarement sur float).
5. **Ce qui est attendu :** Je m'attends à un taux de 5% de faux positifs soit 50 fausses alarmes sur 1000 pièces normales. Pour les faux négatifs en revanche, inconnus. Estimation fragile sur 44 images.
6. **Ce que mesure le test :** Precision / Recall / F1 sur les 58 normales et les 92 anomalies (cat : cable), jamais utilisé pour fixer le seuil.
7. **Angle mort :** Les modèles entraînés sur ces 144 images sont un nouveau run. Les AUROC de référence (224 images, `*_cable_cuda_20261003_171414`) restent la baseline gelée. Non comparables directement aux nouveaux AUROC