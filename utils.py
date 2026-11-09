"""Shared helpers for the animal-shelter ML project.

Everything that more than one notebook (or the dashboard) needs lives here so
training and inference use identical preprocessing, splits, and metrics.

Austin pipeline overview
------------------------
    raw intakes + raw outcomes
        -> clean_intakes / clean_outcomes   (parse dates, dedupe, dogs & cats)
        -> pair_episodes                    (one row per intake "episode")
        -> add_intake_features              (intake-time features only)
        -> add_targets                      (outcome_group, los_days, censoring)
    = build_austin_episodes()  -> data/processed/austin_episodes.parquet

    time_group_split()  -> train / val / test, split by intake date and
                           grouped by Animal ID (see README pitfalls 1-2)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
AUSTIN_INTAKES_PATH = DATA_DIR / "intakes.csv"
AUSTIN_OUTCOMES_PATH = DATA_DIR / "outcomes.csv"
AUSTIN_EPISODES_PATH = PROCESSED_DIR / "austin_episodes.parquet"

ANIMAL_TYPES = ["Dog", "Cat"]

# Raw Austin outcome type -> modeling group. "Other" is kept in the episode
# table but excluded from the 4-class outcome model (it is ~1.5% of dog/cat
# outcomes and mixes unrelated events: died, DOA disposal, missing, stolen).
OUTCOME_GROUP_MAP = {
    "Adoption": "Adoption",
    "Transfer": "Transfer",
    "Return to Owner": "Return to Owner",
    "Rto-Adopt": "Return to Owner",  # owner reclaimed the animal via adoption
    "Euthanasia": "Euthanasia",
    "Died": "Other",
    "Disposal": "Other",
    "Missing": "Other",
    "Lost": "Other",
    "Stolen": "Other",
    "Relocate": "Other",
}
OUTCOME_CLASSES = ["Adoption", "Transfer", "Return to Owner", "Euthanasia"]

# Length-of-stay bins (days). Edges are left-closed: [0, 7), [7, 30), ...
LOS_BIN_EDGES = [0, 7, 30, 90, np.inf]
LOS_BIN_LABELS = ["<1 week", "1-4 weeks", "1-3 months", "3+ months"]

# Stays longer than this are almost always a missing intake/outcome record
# (the animal left and came back without a logged event), not a real stay.
LOS_SUSPECT_DAYS = 365

# Most dog/cat stays finish within this many days (95th pct ~ 85 days). Intakes
# within this window of the end of the data are too recent to have a reliable
# outcome, so they are excluded from supervised train/val/test sets.
CENSOR_HORIZON_DAYS = 90

# Default temporal split boundaries (intake date). Test ends CENSOR_HORIZON_DAYS
# before the last intake in the data; see time_group_split.
VAL_START = "2022-01-01"
TEST_START = "2023-01-01"

# Columns that are known only at outcome time. Never use them as features.
LEAKY_COLUMNS = [
    "outcome_dt", "outcome_type", "outcome_subtype", "sex_outcome", "age_outcome",
    "outcome_group", "los_days", "los_bin", "date_of_birth",
]

# Model-ready intake-time features (built by add_intake_features).
NUMERIC_FEATURES = [
    "age_days", "log_age_days", "n_prior_intakes", "days_since_last_outcome",
    "intake_month", "intake_dayofweek", "intake_hour", "intake_year",
]
BINARY_FEATURES = [
    "is_mix", "is_multicolor", "is_black", "has_name", "is_return",
    "found_in_austin", "found_has_address", "is_male",
]
CATEGORICAL_FEATURES = [
    "animal_type", "intake_type", "intake_condition", "fixed_status",
    "age_group", "breed_primary", "color_primary", "color_pattern",
    "found_city", "season",
]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def load_austin_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw Austin intake and outcome CSVs."""
    intakes = pd.read_csv(AUSTIN_INTAKES_PATH, low_memory=False)
    outcomes = pd.read_csv(AUSTIN_OUTCOMES_PATH, low_memory=False)
    return intakes, outcomes


_AGE_RE = re.compile(r"^\s*(-?\d+)\s+(day|week|month|year)s?\s*$")
_AGE_UNIT_DAYS = {"day": 1, "week": 7, "month": 30.44, "year": 365.25}


def parse_age_to_days(age: pd.Series) -> pd.Series:
    """'2 years' -> 730.5, '3 weeks' -> 21. Negative or unparseable -> NaN."""
    parts = age.astype("string").str.extract(_AGE_RE)
    days = parts[0].astype(float) * parts[1].map(_AGE_UNIT_DAYS)
    return days.where(days >= 0)


def _parse_outcome_datetime(s: pd.Series) -> pd.Series:
    # Outcome timestamps are local wall-clock time; some rows carry a "-05:00"
    # / "-06:00" offset and some don't. Dropping the offset puts every row on
    # the same naive local clock as the intakes table.
    return pd.to_datetime(s.str[:19], format="%Y-%m-%dT%H:%M:%S")


def clean_intakes(df: pd.DataFrame, animal_types=ANIMAL_TYPES) -> pd.DataFrame:
    out = (
        df.drop_duplicates()
        .rename(columns={
            "Animal ID": "animal_id", "Name": "name", "Found Location": "found_location",
            "Intake Type": "intake_type", "Intake Condition": "intake_condition",
            "Animal Type": "animal_type", "Sex upon Intake": "sex_intake",
            "Age upon Intake": "age_intake", "Breed": "breed", "Color": "color",
        })
    )
    out["intake_dt"] = pd.to_datetime(out["DateTime"], format="%m/%d/%Y %I:%M:%S %p")
    out = out.drop(columns=["DateTime", "MonthYear"])
    if animal_types is not None:
        out = out[out["animal_type"].isin(animal_types)]
    return out.sort_values(["animal_id", "intake_dt"]).reset_index(drop=True)


def clean_outcomes(df: pd.DataFrame, animal_types=ANIMAL_TYPES) -> pd.DataFrame:
    out = (
        df.drop_duplicates()
        .rename(columns={
            "Animal ID": "animal_id", "Date of Birth": "date_of_birth",
            "Outcome Type": "outcome_type", "Outcome Subtype": "outcome_subtype",
            "Animal Type": "animal_type_outcome", "Sex upon Outcome": "sex_outcome",
            "Age upon Outcome": "age_outcome",
        })
    )
    out["outcome_dt"] = _parse_outcome_datetime(out["DateTime"])
    out = out[["animal_id", "outcome_dt", "outcome_type", "outcome_subtype",
               "animal_type_outcome", "sex_outcome", "age_outcome", "date_of_birth"]]
    if animal_types is not None:
        out = out[out["animal_type_outcome"].isin(animal_types)]
    return out.sort_values(["animal_id", "outcome_dt"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Intake <-> outcome pairing
# ---------------------------------------------------------------------------
def pair_episodes(intakes: pd.DataFrame, outcomes: pd.DataFrame,
                  late_intake_tolerance: str = "1D") -> tuple[pd.DataFrame, dict]:
    """Return one row per intake episode with its outcome (if any).

    Each outcome is matched to the most recent intake for the same animal at or
    before the outcome time (README pitfall 3). If an animal has several
    outcomes after one intake (no re-intake logged), the first one closes the
    episode. Outcomes with no earlier intake but an intake within
    `late_intake_tolerance` *after* them are timestamp-entry quirks, so they
    are matched to that intake as long as it is still unmatched.

    Intakes with no outcome are kept: their outcome columns are NaN (censored).
    """
    intakes = intakes.copy()
    intakes["episode_id"] = np.arange(len(intakes))
    keys = intakes[["animal_id", "intake_dt", "episode_id"]].sort_values("intake_dt")
    outs = outcomes.sort_values("outcome_dt")

    back = pd.merge_asof(outs, keys, left_on="outcome_dt", right_on="intake_dt",
                         by="animal_id", direction="backward").drop(columns="intake_dt")
    matched = back.dropna(subset=["episode_id"])
    unmatched = back[back["episode_id"].isna()].drop(columns="episode_id")

    fwd = pd.merge_asof(unmatched.sort_values("outcome_dt"), keys, left_on="outcome_dt",
                        right_on="intake_dt", by="animal_id", direction="forward",
                        tolerance=pd.Timedelta(late_intake_tolerance)).drop(columns="intake_dt")
    fwd_ok = fwd.dropna(subset=["episode_id"])
    fwd_ok = fwd_ok[~fwd_ok["episode_id"].isin(matched["episode_id"])]

    all_matched = pd.concat([matched, fwd_ok]).sort_values("outcome_dt")
    n_extra = int(all_matched.duplicated("episode_id").sum())
    all_matched = all_matched.drop_duplicates("episode_id", keep="first")
    all_matched["episode_id"] = all_matched["episode_id"].astype(int)

    episodes = intakes.merge(all_matched.drop(columns="animal_id"), on="episode_id", how="left")
    stats = {
        "intakes": len(intakes),
        "outcomes": len(outcomes),
        "outcomes_matched_backward": len(matched),
        "outcomes_matched_late_intake": len(fwd_ok),
        "outcomes_unmatched": len(outcomes) - len(matched) - len(fwd_ok),
        "extra_outcomes_dropped": n_extra,
        "episodes_without_outcome": int(episodes["outcome_dt"].isna().sum()),
    }
    return episodes, stats


# ---------------------------------------------------------------------------
# Feature engineering (intake-time information only)
# ---------------------------------------------------------------------------
_COLOR_PATTERNS = ["Tabby", "Brindle", "Point", "Tortie", "Calico", "Tricolor",
                   "Merle", "Smoke", "Tiger", "Torbie", "Agouti", "Sable", "Tick"]
_SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
_AGE_BINS = [-0.1, 56, 182, 365, 365 * 3, 365 * 8, np.inf]
_AGE_LABELS = ["<8 weeks", "2-6 months", "6-12 months", "1-3 years", "3-8 years", "8+ years"]


def _primary_breed(breed: pd.Series) -> pd.Series:
    b = breed.str.replace("Black/Tan Hound", "Black-Tan Hound", regex=False)
    return b.str.split("/").str[0].str.replace(r"\s+Mix$", "", regex=True).str.strip()


def _found_city(loc: pd.Series) -> pd.Series:
    city = (loc.str.rsplit(" in ", n=1).str[-1]
               .str.replace(r"\s*\([A-Z]{2}\)\s*$", "", regex=True).str.strip())
    return city.replace("", "Unknown").str.title()


def add_intake_features(ep: pd.DataFrame) -> pd.DataFrame:
    """Add model features derived only from intake-time fields."""
    ep = ep.sort_values(["animal_id", "intake_dt"]).copy()

    # Age
    ep["age_days"] = parse_age_to_days(ep["age_intake"])
    ep["log_age_days"] = np.log1p(ep["age_days"])
    ep["age_group"] = pd.cut(ep["age_days"], _AGE_BINS, labels=_AGE_LABELS).astype(object)

    # Sex / sterilization status at intake
    sex = ep["sex_intake"].fillna("Unknown")
    ep["is_male"] = sex.str.contains("Male").astype(int)
    ep["fixed_status"] = np.select(
        [sex.str.startswith("Intact"), sex.str.startswith(("Neutered", "Spayed"))],
        ["Intact", "Fixed"], "Unknown")

    # Breed & color (raw high-cardinality levels; rare levels are grouped
    # inside the model pipeline by RareLevelGrouper, fit on train only)
    ep["breed_primary"] = _primary_breed(ep["breed"])
    ep["is_mix"] = (ep["breed"].str.contains("Mix") |
                    ep["breed"].str.replace("Black/Tan", "", regex=False).str.contains("/")).astype(int)
    ep["color_primary"] = ep["color"].str.split("/").str[0].str.strip()
    ep["is_multicolor"] = ep["color"].str.contains("/").astype(int)
    pattern = ep["color_primary"].str.extract(f"({'|'.join(_COLOR_PATTERNS)})")[0]
    ep["color_pattern"] = pattern.fillna("Solid")
    ep["is_black"] = (ep["color_primary"] == "Black").astype(int)

    # Name: ~35% of names start with "*", meaning a name assigned by shelter
    # staff. Only an owner-known name counts. NOTE: the export may reflect
    # names updated after intake, so treat this feature with some caution.
    ep["has_name"] = (ep["name"].notna() & ~ep["name"].fillna("").str.startswith("*")).astype(int)

    # Found location
    ep["found_city"] = _found_city(ep["found_location"])
    ep["found_in_austin"] = (ep["found_city"] == "Austin").astype(int)
    ep["found_has_address"] = ep["found_location"].str.contains(" in ", regex=False).astype(int)

    # Intake date
    dt = ep["intake_dt"]
    ep["intake_year"] = dt.dt.year
    ep["intake_month"] = dt.dt.month
    ep["intake_dayofweek"] = dt.dt.dayofweek
    ep["intake_hour"] = dt.dt.hour
    ep["season"] = ep["intake_month"].map(_SEASON)

    # Shelter history for this animal, known at intake: prior episodes and
    # time since the previous episode ended (NaN if first visit).
    g = ep.groupby("animal_id", sort=False)
    ep["n_prior_intakes"] = g.cumcount()
    ep["is_return"] = (ep["n_prior_intakes"] > 0).astype(int)
    prev_outcome = g["outcome_dt"].shift(1)
    ep["days_since_last_outcome"] = (dt - prev_outcome).dt.total_seconds() / 86400
    return ep


def add_targets(ep: pd.DataFrame, data_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Add outcome_group, length of stay, and censoring columns."""
    ep = ep.copy()
    if data_end is None:
        data_end = ep["intake_dt"].max()
    ep["outcome_group"] = ep["outcome_type"].map(OUTCOME_GROUP_MAP)
    ep.loc[ep["outcome_dt"].notna() & ep["outcome_group"].isna(), "outcome_group"] = "Other"

    ep["has_outcome"] = ep["outcome_dt"].notna().astype(int)
    # clip(0): late-logged intakes matched within the 1-day tolerance can give
    # a slightly negative stay
    ep["los_days"] = ((ep["outcome_dt"] - ep["intake_dt"]).dt.total_seconds() / 86400).clip(lower=0)
    # Survival-analysis view: observed duration is LOS if an outcome exists,
    # otherwise time until the end of the data (right-censored).
    ep["duration_days"] = ep["los_days"].fillna(
        (data_end - ep["intake_dt"]).dt.total_seconds() / 86400)
    ep["event_observed"] = ep["has_outcome"]
    ep["los_bin"] = pd.cut(ep["los_days"], LOS_BIN_EDGES, labels=LOS_BIN_LABELS,
                           right=False).astype(object)
    ep["los_suspect"] = (ep["los_days"] > LOS_SUSPECT_DAYS).astype(int)
    ep["is_recent_intake"] = (
        ep["intake_dt"] > data_end - pd.Timedelta(days=CENSOR_HORIZON_DAYS)).astype(int)
    return ep


def build_austin_episodes(save: bool = True) -> tuple[pd.DataFrame, dict]:
    """Full Austin pipeline: raw CSVs -> one cleaned row per intake episode."""
    raw_in, raw_out = load_austin_raw()
    intakes = clean_intakes(raw_in)
    outcomes = clean_outcomes(raw_out)
    ep, stats = pair_episodes(intakes, outcomes)
    data_end = intakes["intake_dt"].max()
    ep = add_intake_features(ep)
    ep = add_targets(ep, data_end=data_end)
    ep = ep.drop(columns=["animal_type_outcome"]).reset_index(drop=True)
    stats["data_end"] = str(data_end)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ep.to_parquet(AUSTIN_EPISODES_PATH, index=False)
    return ep, stats


def load_austin_episodes() -> pd.DataFrame:
    """Load the processed episode table (run eda/austindataprocessing.ipynb first)."""
    if not AUSTIN_EPISODES_PATH.exists():
        raise FileNotFoundError(
            f"{AUSTIN_EPISODES_PATH} not found. Run utils.build_austin_episodes() "
            "or the eda/austindataprocessing.ipynb notebook first.")
    return pd.read_parquet(AUSTIN_EPISODES_PATH)


def model_frame(ep: pd.DataFrame, task: str = "outcome") -> pd.DataFrame:
    """Rows usable for supervised modeling.

    task="outcome": completed episodes in the 4 OUTCOME_CLASSES.
    task="los":     completed episodes with a plausible stay length.
    Both drop intakes too recent to have a reliable outcome and suspect stays.
    """
    keep = (ep["has_outcome"] == 1) & (ep["is_recent_intake"] == 0) & (ep["los_suspect"] == 0)
    if task == "outcome":
        keep &= ep["outcome_group"].isin(OUTCOME_CLASSES)
    elif task != "los":
        raise ValueError("task must be 'outcome' or 'los'")
    return ep[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------
def time_group_split(df: pd.DataFrame, val_start: str = VAL_START,
                     test_start: str = TEST_START, test_end: str | None = None,
                     time_col: str = "intake_dt", group_col: str = "animal_id"):
    """Split by time, keeping every record of an animal on one side.

    Each row gets an era (train/val/test) from its intake date, and each
    animal gets the era of its *first* intake. A row is kept only when the two
    agree, so (a) no animal appears in two splits and (b) no training row is
    from the val/test period. Later returns of train-era animals are dropped
    (they would leak the animal across the split). The drop counts are printed.

    Returns (train, val, test) DataFrames.
    """
    t = df[time_col]
    vs, ts = pd.Timestamp(val_start), pd.Timestamp(test_start)
    era = np.select([t < vs, t < ts], [0, 1], 2)
    first = df.groupby(group_col)[time_col].transform("min")
    animal_era = np.select([first < vs, first < ts], [0, 1], 2)
    keep = era == animal_era
    if test_end is not None:
        keep &= ~((era == 2) & (t >= pd.Timestamp(test_end)))
    n_drop = int((~keep).sum())
    print(f"time_group_split: dropped {n_drop:,} rows ({n_drop / len(df):.1%}) "
          "belonging to animals first seen in an earlier split")
    parts = [df[keep & (era == k)].reset_index(drop=True) for k in range(3)]
    assert not set(parts[0][group_col]) & set(parts[2][group_col])
    return tuple(parts)


# ---------------------------------------------------------------------------
# Preprocessing for scikit-learn models
# ---------------------------------------------------------------------------
class RareLevelGrouper(BaseEstimator, TransformerMixin):
    """Replace categorical levels seen fewer than `min_count` times in the
    training data (or unseen at transform time) with `other`.
    Fit on train only so the level list never peeks at val/test."""

    def __init__(self, min_count: int = 50, other: str = "Other"):
        self.min_count = min_count
        self.other = other

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_ = {c: set(vc[vc >= self.min_count].index)
                      for c in X.columns for vc in [X[c].astype(str).value_counts()]}
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).astype(str)
        for c, keep in self.keep_.items():
            X[c] = X[c].where(X[c].isin(keep), self.other)
        return X

    def get_feature_names_out(self, input_features=None):
        # Upstream steps (e.g. SimpleImputer) pass numpy arrays, so the names
        # seen in fit are 0..n-1; prefer the real names when they're provided.
        names = self.feature_names_in_ if input_features is None else input_features
        return np.asarray(names, dtype=object)


def make_preprocessor(min_count: int = 50, scale_numeric: bool = True,
                      numeric=None, binary=None, categorical=None):
    """ColumnTransformer for FEATURES: impute + (scale) numerics, pass binaries,
    group rare categorical levels then one-hot encode them."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = NUMERIC_FEATURES if numeric is None else numeric
    binary = BINARY_FEATURES if binary is None else binary
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical

    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("rare", RareLevelGrouper(min_count=min_count)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("bin", "passthrough", binary),
        ("cat", cat_pipe, categorical),
    ], verbose_feature_names_out=False)


# ---------------------------------------------------------------------------
# Metrics (use these everywhere so all models are scored identically)
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, labels=None, name: str = "",
                           verbose: bool = True) -> dict:
    """Macro-F1 + per-class precision/recall (README metrics convention)."""
    labels = OUTCOME_CLASSES if labels is None else labels
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                   zero_division=0)
    out = {"model": name, "macro_f1": f1_score(y_true, y_pred, labels=labels,
                                               average="macro", zero_division=0),
           "accuracy": report["accuracy"] if "accuracy" in report
           else float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))}
    for c in labels:
        out[f"precision_{c}"] = report[c]["precision"]
        out[f"recall_{c}"] = report[c]["recall"]
    if verbose:
        print(f"== {name} ==  macro-F1 {out['macro_f1']:.3f}  accuracy {out['accuracy']:.3f}")
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    return out


def regression_metrics(y_true, y_pred, name: str = "", verbose: bool = True) -> dict:
    out = {"model": name,
           "mae": mean_absolute_error(y_true, y_pred),
           "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}
    if verbose:
        print(f"== {name} ==  MAE {out['mae']:.2f}  RMSE {out['rmse']:.2f}")
    return out


def quadratic_weighted_kappa(y_true, y_pred, labels=None) -> float:
    """QWK for ordinal targets (PetFinder AdoptionSpeed, LOS bins).
    If `labels` (in ordinal order) is given, string classes are mapped to ranks."""
    if labels is not None:
        rank = {lab: i for i, lab in enumerate(labels)}
        y_true = pd.Series(y_true).map(rank)
        y_pred = pd.Series(y_pred).map(rank)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# ---------------------------------------------------------------------------
# Shared train-and-evaluate loop for classifiers
# ---------------------------------------------------------------------------
def load_los_splits():
    """Train / val / test for length-of-stay models (target `los_bin` or `los_days`)."""
    df = model_frame(load_austin_episodes(), task="los")
    return time_group_split(df)


def evaluate_classifier(estimator, train: pd.DataFrame, eval_df: pd.DataFrame, name: str,
                        target: str = "los_bin", labels=None, ordinal: bool = True,
                        preprocessor=None, results_dir=None, eval_split: str = "val",
                        extra: dict | None = None):
    """Fit `make_preprocessor() -> estimator` on train, score on eval_df.

    Reports accuracy, macro-F1, per-class precision/recall, and (if ordinal)
    quadratic weighted kappa; plots a row-normalized confusion matrix. If
    `results_dir` is given, saves `<results_dir>/<name>.json` (metrics) and
    `<results_dir>/<name>_confusion.png`, for comparison notebooks to read.
    Returns (fitted_pipeline, metrics_dict).
    """
    import json
    import time

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.pipeline import Pipeline

    labels = LOS_BIN_LABELS if labels is None else labels
    pipe = Pipeline([("pre", make_preprocessor() if preprocessor is None else preprocessor),
                     ("model", estimator)])
    t0 = time.perf_counter()
    pipe.fit(train[FEATURES], train[target])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = pipe.predict(eval_df[FEATURES])
    pred_s = time.perf_counter() - t0
    y_true = eval_df[target]

    metrics = classification_metrics(y_true, y_pred, labels=labels, name=name)
    if ordinal:
        metrics["qwk"] = quadratic_weighted_kappa(y_true, y_pred, labels=labels)
        print(f"quadratic weighted kappa: {metrics['qwk']:.3f}")
    metrics.update({"train_accuracy": float(np.mean(pipe.predict(train[FEATURES]) == train[target])),
                    "fit_seconds": fit_s, "predict_seconds": pred_s,
                    "eval_split": eval_split, "n_train": len(train), "n_eval": len(eval_df),
                    "params": {k: repr(v) for k, v in estimator.get_params().items()}})
    if extra:
        metrics.update(extra)
    print(f"train accuracy {metrics['train_accuracy']:.3f}  |  fit {fit_s:.1f}s  predict {pred_s:.1f}s")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, normalize="true",
                                            values_format=".2f", cmap="Blues", ax=ax,
                                            colorbar=False)
    ax.set_title(f"{name} ({eval_split}, row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        (results_dir / f"{slug}.json").write_text(json.dumps(metrics, indent=2, default=float))
        fig.savefig(results_dir / f"{slug}_confusion.png", dpi=150, bbox_inches="tight")
    plt.show()
    return pipe, metrics


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
TUNE_VAL_START = "2021-01-01"  # inner validation year carved out of the train period


def _to_json(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, tuple):
        return list(v)
    return v


def tune_classifier(estimator, param_distributions: dict, train: pd.DataFrame, name: str,
                    n_iter: int = 15, scoring: str = "f1_macro", target: str = "los_bin",
                    inner_val_start: str = TUNE_VAL_START, n_jobs: int = -1,
                    results_dir=None, verbose: bool = True):
    """Randomized search using a time-ordered inner split *inside* the train period.

    Train rows are split (by time, grouped by animal) into sub-train
    (< inner_val_start) and inner-val (inner_val_start .. end of train). Each
    candidate is fit on sub-train and scored on inner-val; the outer val set
    (2022) and test set are never seen during the search, so the val score
    of the tuned model stays an honest comparison with the untuned one.

    `param_distributions` uses the estimator's own parameter names (no
    `model__` prefix). Returns (best_params, cv_results DataFrame, info dict).
    """
    import time

    from sklearn.base import clone
    from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
    from sklearn.pipeline import Pipeline

    sub, inner, _ = time_group_split(train, val_start=inner_val_start,
                                     test_start="2100-01-01")
    both = pd.concat([sub, inner], ignore_index=True)
    fold = np.r_[np.full(len(sub), -1), np.zeros(len(inner), dtype=int)]

    est = clone(estimator)
    if "n_jobs" in est.get_params():  # parallelize over candidates, not inside each fit
        est.set_params(n_jobs=1)
    pipe = Pipeline([("pre", make_preprocessor()), ("model", est)])
    space = {f"model__{k}": v for k, v in param_distributions.items()}
    n_iter = min(n_iter, _grid_size(param_distributions) or n_iter)

    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring=scoring,
                                cv=PredefinedSplit(fold), refit=False, n_jobs=n_jobs,
                                random_state=SEED, error_score=np.nan)
    t0 = time.perf_counter()
    search.fit(both[FEATURES], both[target])
    secs = time.perf_counter() - t0

    res = pd.DataFrame(search.cv_results_)
    keep = [c for c in res if c.startswith("param_")] + ["mean_test_score", "mean_fit_time"]
    res = (res[keep].rename(columns=lambda c: c.replace("param_model__", ""))
           .rename(columns={"mean_test_score": f"inner_val_{scoring}"})
           .sort_values(f"inner_val_{scoring}", ascending=False).reset_index(drop=True))
    best = {k.replace("model__", ""): _to_json(v) for k, v in search.best_params_.items()}
    if verbose:
        print(f"{name}: {n_iter} candidates in {secs / 60:.1f} min | sub-train {len(sub):,}, "
              f"inner-val {len(inner):,} | best inner-val {scoring} {search.best_score_:.3f}")
        print("best params:", best)
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        res.to_csv(results_dir / f"{slug}_search.csv", index=False)
    info = {"best_params": best, "inner_val_score": float(search.best_score_),
            "tuning_scoring": scoring, "tuning_n_iter": n_iter, "tuning_minutes": secs / 60}
    return best, res, info


def _grid_size(space: dict):
    """Number of combinations if every value is a list, else None (continuous)."""
    n = 1
    for v in space.values():
        if not isinstance(v, (list, tuple)):
            return None
        n *= len(v)
    return n


def with_params(estimator, params: dict):
    """Clone `estimator` and set `params` (from a tuned JSON); lists -> tuples
    for hidden_layer_sizes."""
    from sklearn.base import clone
    p = dict(params)
    if "hidden_layer_sizes" in p and isinstance(p["hidden_layer_sizes"], list):
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
    return clone(estimator).set_params(**p)


# ---------------------------------------------------------------------------
# Estimator workarounds
# ---------------------------------------------------------------------------
from sklearn.neural_network import MLPClassifier as _MLPClassifier  # noqa: E402
from sklearn.preprocessing import LabelEncoder as _LabelEncoder  # noqa: E402


class SafeMLPClassifier(_MLPClassifier):
    """MLPClassifier that accepts string labels with early_stopping=True.

    scikit-learn 1.7's early-stopping validation calls np.isnan on the
    predicted labels, which raises TypeError for string classes such as
    '<1 week'. This subclass trains on integer-encoded labels and decodes
    predictions; everything else is identical to MLPClassifier.
    """

    def fit(self, X, y, **kwargs):
        self._label_encoder = _LabelEncoder().fit(y)
        super().fit(X, self._label_encoder.transform(y), **kwargs)
        self.classes_ = self._label_encoder.classes_
        return self

    def predict(self, X):
        return self._label_encoder.inverse_transform(super().predict(X).astype(int))
