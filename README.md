# Animal Shelter Decision-Support Tool

A data-driven dashboard that helps animal shelters understand which animals are likely to be adopted, how long they will stay, and how many animals the shelter can expect to take in or place. Built as a group project for AI 595 (Applied Machine Learning) and intended to be presented at a research/AI symposium, so it is a **mini research project**, not just a demo.

---

## Vision

A shelter staff member opens the dashboard, enters an animal's details (and uploads a photo), and gets:

1. **Outcome prediction:** how likely is this animal to be adopted, and what is the most likely outcome (adopted, transferred, returned to owner, euthanasia)?
2. **Length-of-stay prediction:** how long is this animal expected to stay, and how fast is it likely to be adopted?
3. **Shelter-level forecasting:** given the animals currently in the shelter plus expected new intakes, how many of each outcome should we expect, and with what uncertainty? (Bayesian methods.)
4. **Insights:** which features matter most (age, breed, vaccine status, intake condition, etc.), and which animal segments are adopted most or least (PCA + clustering).
5. **Photo insights:** does photo quality, photo count, or a visible face relate to faster adoption?

Framing note: outcomes such as euthanasia reflect shelter decisions (often medical/behavioral), not only an animal's "adoptability." The tool should present this output as **"at-risk / needs attention"** to help prioritize resources, not as a judgment about any animal.

---

## Research question

>  What factors predict adoption speed, and can we quantify the effect of photo quality and count on adoption speed?

---

## Datasets

All raw data lives in `data/` and is **gitignored**. Do not commit data files.

| Dataset | File(s) | Used for |
|---|---|---|
| Austin Animal Center Intakes | `data/austin_intakes.csv` | Outcome classification, length of stay, forecasting, segmentation |
| Austin Animal Center Outcomes | `data/austin_outcomes.csv` | Same as above (join with intakes) |
| PetFinder.my Adoption Prediction (Kaggle) | `data/petfinder/` (`train.csv` + images) | Image-quality features, adoption-speed model, photo-count analysis |

Sources:
- Intakes: https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/pyqf-r2dc/about_data
- Outcomes: https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes-10-01-2013-to-05-05-/9t4d-g238/about_data
- PetFinder: https://www.kaggle.com/competitions/petfinder-adoption-prediction/data

### Important: the datasets do NOT link to each other

- Austin data has **no images**. PetFinder has images but **no Austin outcomes**. (PetFinder is Malaysian listings from ~2018.)
- Therefore: **Austin** powers outcome, length-of-stay, forecasting, and segmentation. **PetFinder** powers image-quality and adoption-speed modeling. They are separate model branches, combined only at the dashboard level.
- Austin likely has no vaccine-status column !! (verify against the actual columns). PetFinder has `Vaccinated`, `Dewormed`, `Sterilized`, `Health`, `PhotoAmt`, etc.
- PetFinder target `AdoptionSpeed` is ordinal: 0 = same day, 1 = 1-7 days, 2 = 8-30 days, 3 = 31-90 days, 4 = not adopted after 100 days.

---

## Modeling tasks

| Task | Data | Type | Models (from class) |
|---|---|---|---|
| Outcome prediction | Austin | Multi-class classification (adopted / transferred / return to owner / euthanasia) | Logistic regression, decision tree, random forest / gradient boosting |
| Length of stay | Austin | Regression or binned classification | Linear regression, random forest |
| Segmentation | Austin | Unsupervised | PCA (visualization), k-means / GMM |
| Adoption speed | PetFinder | Ordinal / multi-class classification | Tree ensembles, logistic regression |
| Image quality | PetFinder images | Handcrafted features (blur, brightness, resolution, face detected) and/or CNN embeddings | Pretrained CNN features, small neural net |
| Forecasting | Austin | Bayesian | Per-animal outcome probabilities aggregated into count distributions; Poisson / negative-binomial intake model with seasonality (e.g. PyMC) |

The course emphasizes applying analysis and **tuning the models taught in class** (linear/logistic regression, trees, ensembles, Bayesian learning, neural networks, unsupervised learning, intro deep learning). Prefer these over exotic methods. Do not train LLMs.

### Scope tiers
- **Core:** Austin outcome classifier + length-of-stay model, feature importance, Streamlit dashboard.
- **Target:** PetFinder adoption-speed model with image-quality features + ablation; PCA/cluster segmentation.
- **Reach:** Bayesian intake/outcome forecasting, survival analysis for censored stays, CNN embeddings, subgroup fairness checks (e.g. by breed or color).

---

## Repo structure

```
animal-shelter-ml/
├── README.md
├── data/
│   ├── austin_intakes.csv
│   ├── austin_outcomes.csv
│   └── petfinder/            # train.csv + images
├── eda/                      # explore all three datasets
├── outcome_pred/             # adopted / transferred / RTO / euthanasia
├── length_of_stay/
├── PCA/                      # segmentation + clustering
├── image_model/              # image quality + adoption speed (PetFinder)
├── forecasting/              # Bayesian forecasting
├── dashboard/                # Streamlit app
└── utils.py                  # shared: data loading, train/test split, metrics
```

Each analysis folder contains numbered notebooks (`01_clean.ipynb`, `02_model.ipynb`, ...) plus any saved figures or models it produces.

---

## Conventions

- **Language/stack:** Python, pandas, NumPy, scikit-learn, matplotlib/seaborn, PyMC (Bayesian), PyTorch/torchvision (images), Streamlit (dashboard).
- **Shared code goes in `utils.py`.** If a function is used in more than one notebook, move it there. Notebooks should call shared helpers, not copy-paste them. The train/test split and metric functions must live in `utils.py` so every model is evaluated identically.
- **Notebook naming:** `NN_verb_noun.ipynb`, numbered in run order within a folder. No `final_v2` files; use git history.
- **Paths:** use relative paths from the repo root. No absolute or machine-specific paths.
- **Randomness:** fixed seed (`SEED = 42`) defined in `utils.py`.
- **Metrics:** report macro-F1 and per-class precision/recall for outcome classification (not just accuracy). For PetFinder adoption speed, also report quadratic weighted kappa. For regression, report MAE and RMSE.
- **Baselines first:** always compare against a simple baseline (majority class / mean) before tuning.
- **Notebook hygiene:** clear outputs before committing (`nbstripout`), and save key figures to the relevant analysis folder.

---

## Pitfalls

1. **Leakage in Austin data.** Do not use outcome-time fields as features: `Outcome Subtype`, `Sex upon Outcome` (spay/neuter often happens at the shelter before adoption), or `Age upon Outcome`. Use intake-time features (`Sex upon Intake`, `Age upon Intake`, `Intake Condition`, `Intake Type`, `Animal Type`, `Breed`, `Color`, found location, intake date/season).
2. **Split by time, grouped by animal.** Train on earlier years and test on later ones, and keep all records for an `Animal ID` on one side of the split. Random row-wise splits leak because animals can have repeat stays.
3. **Joining intakes and outcomes.** Animals can have multiple intake/outcome pairs. Match each outcome to the most recent preceding intake for the same `Animal ID`, not a naive merge on ID.
4. **Class imbalance.** Euthanasia (and some other outcomes) is rare. Use class weights or resampling, and judge models on macro-F1 / PR-AUC.
5. **Censoring.** Animals still in the shelter have no outcome yet. Regression on completed stays only is biased toward short stays; note this or use survival analysis.
6. **High-cardinality categoricals.** Breed and color have many levels. Group rare levels, extract primary breed / mixed flag, and avoid blowing up one-hot features for PCA.
7. **Don't merge Austin and PetFinder rows.** See the dataset section.
8. **Interpretation.** Feature importance and cluster differences are associations, not causes. Word results accordingly.



## Dashboard (Streamlit)

Pages:
- **Predict:** enter animal info + upload photo, show outcome probabilities, expected length of stay, and a photo-quality/adoption-speed score.
- **Insights:** feature importance, top and bottom adopted segments, trends by breed/age/intake condition.
- **Forecast:** expected outcome counts for the current shelter population with uncertainty intervals.

Models are trained in notebooks, saved (e.g. `joblib` / `.pt`), and loaded by the dashboard. The dashboard should import shared code from `utils.py` so training and inference preprocessing stay identical.


## Setup

```bash
git clone <repo-url>
cd animal-shelter-ml
pip install -r requirements.txt   # or conda env create -f environment.yml
# download the three datasets into data/ (see Datasets above)
streamlit run dashboard/app.py
```
