# Project context for the write-up

This is a briefing for an assistant helping write and edit the research write-up (LaTeX, in Overleaf) for this project. The code lives in the GitHub repo `annahauk/adoptionML`, and this file summarizes it. **Every number below comes from code that was actually run.** If a number isn't here or in a synced repo file, it doesn't exist yet: do not invent it.

Last updated: 2026-11-09. If the repo files disagree with this document, trust the repo.

---

## 1. Writing rules (read first)

1. **Never invent numbers, citations, or results.** If a sentence needs a value that isn't in this file, write `\todo{value from X}` (or `[TODO: ...]`) and keep going.
2. **Associations, not causes.** Feature importance, outcome-rate differences and cluster differences are associations. Use "is associated with" and "predicts", never "causes" or "leads to".
3. **Euthanasia framing.** Euthanasia is mostly a medical or behavioral shelter decision, not a verdict on an animal's adoptability. In prose and figures, present that class as **"at-risk / needs attention"**.
4. **Metric names stay fixed:** accuracy, macro-F1, quadratic weighted kappa (QWK), precision, recall, MAE, RMSE. Always say which split a number is on (val = 2022; test is still untouched).
5. **Always give the baseline** next to any accuracy. Accuracy is misleading here because the classes are imbalanced.
6. **LaTeX:** produce compilable LaTeX. Use `booktabs` for tables (`\toprule`, `\midrule`, `\bottomrule`), `\cref`/`\ref` for cross-references, and `\cite{key}` with a BibTeX entry for any citation. Don't change the document class or preamble unless asked.
7. **Tone:** a course research project (AI 595, Applied Machine Learning) headed for a research/AI symposium. Clear and concise, honest about limits. Limitations are part of the contribution, not something to hide.

---

## 2. Project in one paragraph

A decision-support tool for animal shelters. From an animal's details at intake, it predicts (a) the likely outcome (adoption, transfer, return to owner, euthanasia), (b) the expected length of stay, and later (c) shelter-level forecasts and (d) whether photo quality relates to adoption speed. It is presented through a Streamlit dashboard.

**Research question:** *What factors predict adoption speed, and can we quantify the effect of photo quality and count on adoption speed?*

## 3. Data

| Dataset | Use | Notes |
|---|---|---|
| Austin Animal Center Intakes + Outcomes (City of Austin open data, Oct 2013 – May 2025) | Outcome prediction, length of stay, (planned) forecasting and segmentation | No images. ~173.8k intake rows and ~173.8k outcome rows in total, all species. |
| PetFinder.my Adoption Prediction (Kaggle, Malaysia, ~2018) | (Planned) photo-quality and adoption-speed analysis | Has images and `AdoptionSpeed` (0–4 ordinal). **Not linked to Austin; the two are never merged.** |

Scope for all Austin modeling: **dogs and cats only.**

## 4. Data preparation (Austin)

Code: `utils.py` (`build_austin_episodes`). Notebook: `eda/austindataprocessing.ipynb`.

- **Unit of analysis = intake episode** (one stay of one animal): **163,901 episodes**. Animals can have many stays; one dog has dozens.
- **Joining:** a naive merge on Animal ID is many-to-many and created **47,346 spurious rows**. Instead, each outcome is matched to the **most recent intake at or before it** for the same animal (`merge_asof`). That matches **99.4%** of dog/cat outcomes (162,437 directly, plus 524 where the intake was logged up to 1 day after the outcome). If one intake has several outcomes, the first one counts.
- **Timestamps:** outcome times carry inconsistent UTC offsets. They are normalized to local wall-clock time to match the intakes.
- **Outcome labels:** Adoption, Transfer, Return to Owner (includes "Rto-Adopt"), Euthanasia; plus "Other" (died, disposal, missing, stolen, lost, relocate; ~1%), which is excluded from the 4-class model.
- **Leakage rule:** only **intake-time** information is used as features. Never used: Sex upon Outcome, Age upon Outcome, Outcome Subtype, outcome date. Evidence for the rule: 73% of adopted animals were spayed or neutered *during* their stay, so sex-upon-outcome partly encodes the outcome.
- **Features (26):**
  - age in days (and its log), age group;
  - species, intake type, intake condition;
  - sterilization status at intake, sex;
  - primary breed and mixed-breed flag; primary color, color pattern, multicolor flag, black-coat flag;
  - owner-given name flag (`has_name`);
  - number of prior intakes, return flag, days since last outcome;
  - found city, found inside Austin, found address given;
  - intake year, month, day of week, hour, season.

  Rare breed, color and city levels are grouped into "Other" using **training data only**.
- **Censoring:** 1,019 intakes have no outcome (740 of them from the last 90 days). Intakes within 90 days of the end of the data are excluded from supervised train/eval, so the most recent data isn't biased toward short stays. Stays over 365 days (568; almost always a missing record) are excluded.
- **Split: by time and grouped by animal.** Train: intake < 2022-01-01. Val: 2022. Test: 2023-01-01 to 2025-02-03, **not used yet**. Each animal belongs to the split of its first intake, so no animal appears in two splits; about 0.8% of rows are dropped as a result. A random row-wise split would leak repeat animals and hide drift.

| Split | Outcome task rows | Length-of-stay task rows |
|---|---|---|
| Train | 125,490 | 126,758 |
| Val | 10,430 | 10,544 |
| Test (held out) | 21,463 | 21,835 |

## 5. EDA findings (dogs + cats, associations)

- **Outcome mix:** Adoption 51%, Transfer 29%, Return to Owner 16%, Euthanasia 3%, Other 1%.
- **Species differ sharply:** dogs return to owner 25% vs cats 4.5%; cats transfer 39% vs dogs 22%.
- **Intake condition:** Injured / Sick / Aged → 15–18% euthanasia, vs 1.5% for Normal. Nursing / Neonatal → 65–71% transfer, mostly to foster.
- **Euthanasia subtypes:** Suffering 68%, Aggressive + Behavior ~16%, Rabies risk ~5%. **None are space-related.**
- **Sterilized at intake** → return to owner 34% vs 11% for intact. "Unknown" sex is mostly animals under 8 weeks and community cats (trap-neuter-return programs) → 87% transfer.
- **Repeat visitors** (`is_return`) → return to owner 30% vs 15%.
- **Length of stay:** median 6.2 days, 75th percentile 21 days, 95th percentile 87 days. Median stay for adopted animals: cats 25 days, dogs 9 days. The age effect is **U-shaped**: neonates wait in foster until old enough for sterilization and adoption (cats under 8 weeks: median 42 days to adoption), and seniors wait longest among adults (cats 8+ years: 27 days).
- **Black coat:** black cats are adopted slightly less (50.8% vs 52.8%) and stay about 1 day longer; black dogs are adopted slightly *more* (52.6% vs 49.7%). The effect is small and mixed.
- **Seasonality:** cat intakes peak in May–June at about 1.5× the monthly average (kitten season). Dog intakes are flat across the year. Intakes dropped sharply in 2020 (COVID).
- **Drift:** adoption share rose from about 42% (2013–15) to 65% (2024); return to owner fell from about 20% to 9%; euthanasia fell from 6.7% (2013) to about 2%.

## 6. Results so far (all on val = 2022; default hyperparameters unless noted)

### 6a. Outcome prediction (4 classes), `eda/` baselines and quick checks

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class baseline | 0.604 | 0.188 |
| Logistic regression (class-balanced) | 0.609 | 0.468 |
| Gradient boosting (HistGradientBoosting) | 0.686 | 0.490 |
| Gradient boosting **without** `has_name` | 0.670 | 0.454 |

Full model comparison for outcome prediction (`outcome_pred/`): **not done yet**.

### 6b. Length of stay as 4 ordinal bins (<1 week, 1–4 weeks, 1–3 months, 3+ months), `length_of_stay/`

Bin shares: train 56 / 26 / 14 / 4%; val 43 / 32 / 19 / 5% (stays got longer after 2021).

| Model | Accuracy | Macro-F1 | QWK | Train acc. |
|---|---|---|---|---|
| Gradient boosting | **0.528** | 0.360 | 0.301 | 0.638 |
| Random forest | 0.496 | 0.337 | 0.249 | 0.994 |
| Neural network (MLP, 1×100) | 0.486 | **0.375** | **0.310** | 0.674 |
| Extra trees | 0.484 | 0.328 | 0.234 | 0.994 |
| Logistic regression | 0.462 | 0.297 | 0.179 | 0.574 |
| Linear SVM | 0.459 | 0.273 | 0.135 | 0.573 |
| k-nearest neighbors (k=5) | 0.435 | 0.331 | 0.207 | 0.707 |
| *Majority class baseline* | *0.432* | *0.151* | *0.000* | *0.559* |
| AdaBoost (50 stumps) | 0.425 | 0.228 | 0.048 | 0.548 |
| Decision tree | 0.416 | 0.321 | 0.172 | 0.994 |
| Gaussian naive Bayes | 0.158 | 0.154 | 0.080 | 0.177 |

Recall for "3+ months" is ≤ 9% for every real model. Without `has_name`, gradient boosting gets accuracy 0.507 and macro-F1 0.339.

### 6c. Why length of stay is hard: diagnostics (gradient boosting)

| Experiment | Accuracy | Macro-F1 | QWK | "3+ months" recall |
|---|---|---|---|---|
| Reference: train < 2022 → val 2022 | 0.528 | 0.360 | 0.301 | 0.01 |
| Same-era random holdout (2013–21, no drift); its baseline is 0.555 | 0.619 | 0.391 | 0.350 | 0.02 |
| Train only on 2019–2021 | 0.525 | 0.366 | 0.322 | 0.01 |
| `class_weight="balanced"` | 0.448 | 0.403 | 0.312 | 0.44 |
| Oracle: true outcome given as a feature (leaky, diagnostic only) | 0.598 | 0.435 | 0.532 | 0.02 |

- Even knowing the eventual outcome only lifts accuracy to about 0.60, so most of the variation in stay length isn't knowable at intake (medical holds, foster availability, adopter demand).
- 21.5% of val stays fall within ±2 days of a bin edge (7 / 30 / 90 days), which makes the labels noisy.
- Animals that look identical have very different stays. Adult, healthy stray cats in 2022 spread 47 / 34 / 16 / 3% across the four bins.
- Drift lowers raw accuracy but not the improvement over the baseline: +9.6 points on val vs +6.4 points on the same-era holdout.

**Interpretation for the paper:** length of stay is only weakly predictable from intake data. That's a finding, not a failure. Report the improvement over the baseline and QWK. Class weighting trades accuracy for finding long-stay animals, which is the operationally useful direction.

### 6d. Length of stay after hyperparameter tuning (`length_of_stay/` notebooks 02–12)

**Tuning procedure** (`utils.tune_classifier`):
- Randomized search, 10–25 candidates per model.
- A time-ordered inner split inside the training years: candidates fit on intakes before 2021, scored on 2021 intakes (grouped by animal).
- Candidates are ranked by **macro-F1**. Class weighting is part of the search space wherever the model supports it.
- The best setting is refit on all training years (< 2022) and scored on the same 2022 val set as the untuned models. Val and test are never used for selection.

| Model | Untuned acc / macro-F1 / QWK | Tuned acc / macro-F1 / QWK | Tuned "3+ months" recall | Tuned train acc |
|---|---|---|---|---|
| Random forest | 0.496 / 0.337 / 0.249 | 0.483 / **0.405** / **0.342** | 0.16 | 0.798 |
| Extra trees | 0.484 / 0.327 / 0.234 | 0.487 / 0.384 / 0.313 | 0.11 | 0.960 |
| Gradient boosting | **0.528** / 0.360 / 0.301 | 0.517 / 0.383 / 0.314 | 0.07 | 0.690 |
| Decision tree | 0.416 / 0.321 / 0.172 | 0.423 / 0.374 / 0.245 | 0.31 | 0.443 |
| Linear SVM | 0.459 / 0.273 / 0.135 | 0.422 / 0.370 / 0.268 | 0.44 | 0.518 |
| Neural network (MLP) | 0.486 / 0.375 / 0.310 | 0.514 / 0.364 / 0.298 | 0.02 | 0.642 |
| Logistic regression | 0.462 / 0.297 / 0.179 | 0.385 / 0.360 / 0.217 | 0.57 | 0.445 |
| AdaBoost | 0.425 / 0.228 / 0.048 | 0.505 / 0.348 / 0.258 | 0.01 | 0.588 |
| k-nearest neighbors | 0.435 / 0.331 / 0.207 | 0.462 / 0.322 / 0.206 | 0.01 | 0.643 |
| Gaussian naive Bayes | 0.158 / 0.154 / 0.080 | 0.333 / 0.274 / 0.067 | 0.09 | 0.423 |
| *Majority baseline* | *0.432 / 0.151 / 0.000* | — | — | — |

- **Best tuned model: random forest** (macro-F1 0.405, QWK 0.342). Settings: `class_weight="balanced"`, 300 trees, `max_depth=30`, `min_samples_leaf=5`, `max_features=0.3`. Train accuracy fell from 0.994 to 0.798, so it overfits much less.
- The search chose `class_weight="balanced"` for every model that supports it except gradient boosting. This trades accuracy for recall on long stays: for example, logistic regression's accuracy drops to 0.385, but its "3+ months" recall rises from 0.01 to 0.57. Rankings by accuracy and by macro-F1 differ, so always report both.
- MLP and k-NN got slightly worse on macro-F1 after tuning, because settings chosen on 2021 don't fully carry over to 2022 (drift). Report this as is.
- The ceiling is about 0.40 macro-F1 and 0.34 QWK, consistent with §6c.
- Implementation note: `utils.SafeMLPClassifier` works around a scikit-learn 1.7 bug (`MLPClassifier` with `early_stopping=True` fails on string labels). It doesn't change the model.

### 6e. One combined model vs. separate dog and cat models (`length_of_stay/13_dogs_vs_cats.ipynb`)

Same split and tuned settings. "Species-specific" means a dog-only model for dogs and a cat-only model for cats; the dog training set is about 75k rows and the cat set about 51k. Δ = species-specific − combined, macro-F1 on val 2022:

| Model | Δ all | Δ dogs | Δ cats |
|---|---|---|---|
| Random forest | −0.008 | +0.006 | −0.008 |
| Extra trees | −0.004 | −0.002 | −0.002 |
| Gradient boosting | −0.015 | −0.010 | −0.013 |
| Neural network (MLP) | −0.009 | +0.001 | −0.015 |
| Logistic regression | −0.002 | +0.013 | +0.010 |
| Linear SVM | +0.022 | +0.032 | +0.008 |
| Gaussian naive Bayes | +0.044 | +0.014 | +0.060 |

- Gradient boosting **re-tuned separately per species** reaches overall macro-F1 0.363, vs 0.383 for the combined model.
- **Conclusion:** splitting by species does not help the strong models. Changes are within about ±0.015 macro-F1, and gradient boosting and the MLP get worse on cats (gradient boosting's cat accuracy goes 0.506 → 0.479). Only the weak models benefit, because they can't model species interactions themselves.
- **Recommendation:** use one combined model with `animal_type` as a feature, and report per-species metrics as a subgroup check. Cats are slightly harder to predict than dogs for every model.

## 7. Decisions and caveats

- **`has_name` is kept as a feature (team decision).** Evidence that it may leak: the Name field is one current-name value copied onto every record. It is identical across all of an animal's stays and between intake and outcome rows. Among strays, 93.5% of those returned to their owner have a name vs 16% of those transferred. The write-up should state this caveat and report results with and without the feature (numbers in 6a and 6b).
- **The test set is untouched.** All numbers so far are on val. Test is scored once, after tuning.
- **Tuning is light to medium** (10–25 random candidates per model, one inner validation year). A larger search or time-series cross-validation over several years could change the rankings slightly. Untuned deep trees overfit (99% train accuracy); tuning reduced that (random forest: 80%).
- **Species:** one combined model is used (see 6e).
- The prior-intake count misses any history before Oct 2013.

## 8. Related work notes

- **Thakur, Chauhan & Pundir, "Machine Learning Model To Predict the Pet Adoption Likelihood", ICACCTech 2024, DOI 10.1109/ICACCTech65084.2024.00125.** Reports AdaBoost at 91.54% (SVM 88%, logistic regression 87%). Not comparable to our results, and methodologically weak:
  - The dataset is a ~2,008-row Kaggle file (cats, dogs, birds, rabbits), likely the "Predict Pet Adoption Status" dataset; check whether it is synthetic before saying so in print.
  - It uses **Time in Shelter as a feature**, which is outcome-time information.
  - The paper is inconsistent about its target (abstract: adoption speed; SVM section: adopted vs not adopted; logistic regression section: five categories).
  - Its own learning curves show validation accuracy of about 0.67 / 0.71 / 0.72, not 88–91%.
  - Random 80/20 split, ordinal label-encoding of nominal categories, no baseline, no macro-F1.

  Use it as a cautionary example of leakage, a random split and no baseline.
- **"Parte's work"** (cited in the paper above): Austin Animal Center outcomes, XGBoost, **65.33%** accuracy. The closest prior result on our data; split not described.
- **Bradley & Rajendran (2021), BMC Veterinary Research:** a two-phase approach to predicting length of stay and shelter allocation. Relevant to our length-of-stay work.
- **IEEE doc 10019608:** "Predicting Animal Shelter Pet Adoption Times and Feature Importance Analysis using CatBoost". Uses Austin data 2013–Jul 2022 with binary adoption time (<15 vs >15 days). Close to our planned binary long-stay model. Check its split method before comparing.
- **PetFinder competition:** top entries reach only about 0.45 QWK on 5-class AdoptionSpeed, so very high accuracy on that task signals leakage.

Get exact BibTeX from the publisher or Google Scholar; don't reconstruct it from memory.

## 9. Status and planned work

- Done: Austin cleaning and join, EDA, leakage-safe features, time-grouped split, length-of-stay model comparison (untuned and tuned), dogs-vs-cats comparison, outcome baselines.
- Next:
  - outcome-prediction model comparison (`outcome_pred/`);
  - final test-set evaluation (once) of the shortlisted length-of-stay models: tuned random forest and gradient boosting;
  - a binary long-stay (≥ 30 days) model, and/or regression on log-days, and/or survival analysis;
  - permutation feature importance.
- Later: PetFinder photo-quality features and ablation; PCA and clustering segmentation; Bayesian intake forecasting (seasonal); Streamlit dashboard; a subgroup check (breed, color).

## 10. Suggested paper outline

1. Introduction (problem, research question, contributions)
2. Related work (incl. the leakage critique in §8)
3. Data (both datasets; why they aren't merged)
4. Methods (episode construction, leakage rule, time-grouped split, censoring, models, metrics)
5. Results (EDA highlights; outcome model; length-of-stay comparison; diagnostics)
6. Discussion and limitations (predictability ceiling, drift, `has_name`, associations not causes, euthanasia framing)
7. Future work




## Why bad inital model performance:
The models aren't badly built. Length of stay simply can't be predicted well from what's known at intake, and that sets a low ceiling. A few other factors pull scores down further.

1. The information isn't in the intake data (the biggest factor). Two checks show this:

Identical-looking animals have very different stays. Take adult, healthy stray cats in 2022: 47% left within a week, 34% in 1–4 weeks, 16% in 1–3 months and 3% after 3+ months. Nothing at intake separates them.
Even knowing the future barely helps. As a diagnostic only, I gave the model each animal's eventual outcome (adopted, returned, etc.), which is information it could never have in practice. Accuracy only rose from 0.528 to 0.598. Most of the variation comes from things that happen during the stay: medical holds, foster availability, how quickly an adopter shows up.
2. The bins add noise. 21.5% of 2022 stays fall within 2 days of a bin edge (7, 30 or 90 days). A 6-day stay and an 8-day stay are almost the same, but they count as different classes, so a near-miss scores the same as a big miss. QWK gives partial credit for near misses, which is why I report it alongside accuracy.

3. Imbalance, and no class weights. "3+ months" is 4% of training data, so models learn to almost never predict it (1% recall). Balancing the class weights moves the trade-off:

Accuracy	Macro-F1	"3+ months" recall
Default	0.528	0.360	0.01
class_weight="balanced"	0.448	0.403	0.44
Accuracy drops, but the model starts flagging long-stay animals, which is what a shelter actually needs.

4. Drift lowers the raw numbers, but not the improvement over the baseline. When I test inside the training years with no time gap, accuracy is 0.619. That holdout's always-"under 1 week" baseline is 0.555, though, so the lift is only 6.4 points. On 2022 the lift is 9.6 points. Stays got longer after 2021, which lowers every score, but the model is no worse relative to the baseline. Training only on 2019–2021 didn't help either (0.525).

5. No tuning yet. The deep trees memorize the training data (99% train accuracy), and the linear models are too simple. Tuning will help, but probably by a few points, not dozens, given point 1.

What I'd do next

For the write-up: treat "length of stay is mostly not predictable at intake" as a finding. Report the improvement over the baseline and QWK, not raw accuracy alone.
Turn on class weights and judge the models on macro-F1 and QWK. The useful question for staff is "is this animal at risk of a long stay?", and that needs the "3+ months" bin to be found.
Consider changing the target:
a binary "long stay (30+ days)?" classifier, which is easier and more useful;
regression on log(days);
survival analysis, which also handles animals still in the shelter.
Tune gradient boosting, the MLP and random forest, as planned.
