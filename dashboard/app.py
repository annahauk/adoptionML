"""Streamlit dashboard for Austin Animal Center decision support.

The repository does not currently contain serialized model artifacts. Until
those artifacts are produced, this app uses transparent historical baselines
computed from the Austin intake/outcome files and labels them as estimates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
INTAKES_PATH = ROOT / "data" / "intakes.csv"
OUTCOMES_PATH = ROOT / "data" / "outcomes.csv"
OUTCOME_ORDER = ["Adoption", "Transfer", "Return to Owner", "Euthanasia", "Other"]
OUTCOME_COLORS = {
    "Adoption": "#2e7d32",
    "Transfer": "#1565c0",
    "Return to Owner": "#ef6c00",
    "Euthanasia": "#c62828",
    "Other": "#616161",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file was not found: {path}")
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner="Loading shelter records…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data and construct one leakage-safe historical training table."""
    intakes = _read_csv(INTAKES_PATH)
    outcomes = _read_csv(OUTCOMES_PATH)
    intakes["intake_date"] = pd.to_datetime(
        intakes["DateTime"], errors="coerce", format="mixed"
    )
    outcomes["outcome_date"] = pd.to_datetime(
        outcomes["DateTime"], errors="coerce", utc=True, format="mixed"
    ).dt.tz_localize(None)

    intake_for_join = intakes[
        ["Animal ID", "intake_date", "Intake Type", "Intake Condition",
         "Animal Type", "Sex upon Intake", "Age upon Intake", "Breed", "Color"]
    ].dropna(subset=["Animal ID", "intake_date"]).sort_values(["intake_date", "Animal ID"])
    outcome_for_join = outcomes[
        ["Animal ID", "outcome_date", "Outcome Type", "Outcome Subtype"]
    ].dropna(subset=["Animal ID", "outcome_date"]).sort_values(["outcome_date", "Animal ID"])

    joined = pd.merge_asof(
        outcome_for_join,
        intake_for_join,
        left_on="outcome_date",
        right_on="intake_date",
        by="Animal ID",
        direction="backward",
    )
    joined["length_of_stay"] = (
        joined["outcome_date"] - joined["intake_date"]
    ).dt.total_seconds().div(86400)
    joined = joined[joined["length_of_stay"].between(0, 3650)].copy()
    joined["Outcome Group"] = joined["Outcome Type"].map(normalize_outcome)
    return intakes, outcomes, joined


def normalize_outcome(value: Any) -> str:
    value = str(value).strip().lower()
    if "adopt" in value:
        return "Adoption"
    if "transfer" in value:
        return "Transfer"
    if "return" in value or value in {"rto", "returned to owner"}:
        return "Return to Owner"
    if "euth" in value:
        return "Euthanasia"
    return "Other"


def _options(frame: pd.DataFrame, column: str, fallback: str) -> list[str]:
    values = (
        frame[column].dropna().astype(str).str.strip().replace("", np.nan).dropna()
        .value_counts().head(80).index.tolist()
    )
    return values or [fallback]


def _filtered_history(
    joined: pd.DataFrame, animal_type: str, condition: str, intake_type: str
) -> pd.DataFrame:
    result = joined[joined["Animal Type"].eq(animal_type)]
    if condition != "Any":
        result = result[result["Intake Condition"].eq(condition)]
    if intake_type != "Any":
        result = result[result["Intake Type"].eq(intake_type)]
    return result


def _probabilities(history: pd.DataFrame) -> pd.Series:
    counts = history["Outcome Group"].value_counts().reindex(OUTCOME_ORDER, fill_value=0)
    if counts.sum() == 0:
        return pd.Series(1 / len(OUTCOME_ORDER), index=OUTCOME_ORDER)
    return counts / counts.sum()


def _photo_summary(upload: Any) -> dict[str, Any] | None:
    if upload is None:
        return None
    try:
        from PIL import Image, ImageStat

        image = Image.open(upload)
        brightness = float(np.mean(ImageStat.Stat(image.convert("L")).mean))
        width, height = image.size
        return {
            "width": width,
            "height": height,
            "brightness": brightness,
            "quality": "Good" if width >= 500 and 35 <= brightness <= 225 else "Review",
        }
    except (ImportError, OSError, ValueError) as error:
        st.warning(f"Photo quality could not be measured: {error}")
        return None


def render_predict(intakes: pd.DataFrame, joined: pd.DataFrame) -> None:
    st.header("Predict")
    st.caption(
        "A transparent historical baseline for prioritization. It is not a "
        "clinical or adoption-worthiness judgment."
    )
    left, right = st.columns([1, 1])
    with left:
        animal_type = st.selectbox("Animal type", _options(intakes, "Animal Type", "Dog"))
        age = st.number_input("Age (years)", min_value=0.0, max_value=30.0, value=2.0, step=0.5)
        sex = st.selectbox("Sex upon intake", _options(intakes, "Sex upon Intake", "Unknown"))
        breed = st.text_input("Breed", value="Unknown")
        condition = st.selectbox(
            "Intake condition", ["Any"] + _options(intakes, "Intake Condition", "Normal")
        )
        intake_type = st.selectbox(
            "Intake type", ["Any"] + _options(intakes, "Intake Type", "Stray")
        )
    with right:
        photo = st.file_uploader("Upload a photo (optional)", type=["jpg", "jpeg", "png"])
        photo_info = _photo_summary(photo)
        if photo is not None:
            st.image(photo, caption="Uploaded photo", use_container_width=True)
        if photo_info:
            st.metric("Photo quality signal", photo_info["quality"])
            st.caption(
                f"{photo_info['width']} × {photo_info['height']} px · "
                f"brightness {photo_info['brightness']:.0f}/255"
            )
        st.info(
            "Photo quality is a screening signal only. A trained PetFinder image "
            "model can replace this baseline when its artifacts are available."
        )

    history = _filtered_history(joined, animal_type, condition, intake_type)
    probabilities = _probabilities(history)
    expected_stay = history["length_of_stay"].median() if not history.empty else np.nan
    if pd.isna(expected_stay):
        expected_stay = joined["length_of_stay"].median()

    st.subheader("Estimated outcome mix")
    metric_columns = st.columns(4)
    for column, label in zip(metric_columns, OUTCOME_ORDER[:4]):
        column.metric(label, f"{probabilities[label]:.0%}")
    st.caption(f"Based on {len(history):,} comparable completed stays.")

    chart = probabilities.rename("Probability").to_frame()
    st.bar_chart(chart, horizontal=True, color="#2e7d32")
    stay_col, note_col = st.columns([1, 2])
    stay_col.metric("Expected length of stay", f"{expected_stay:.1f} days")
    note_col.warning(
        "Use this output to prioritize enrichment, medical review, or follow-up. "
        "Historical outcomes reflect shelter decisions and resource constraints."
    )


def render_insights(intakes: pd.DataFrame, joined: pd.DataFrame) -> None:
    st.header("Insights")
    st.caption("Descriptive associations from completed Austin shelter records.")
    animal_type = st.selectbox("Segment by animal type", ["All"] + _options(intakes, "Animal Type", "Dog"))
    frame = joined if animal_type == "All" else joined[joined["Animal Type"].eq(animal_type)]
    if frame.empty:
        st.warning("There are not enough records for this selection.")
        return

    adoption_rate = frame["Outcome Group"].eq("Adoption").mean()
    metric_columns = st.columns(3)
    metric_columns[0].metric("Completed stays", f"{len(frame):,}")
    metric_columns[1].metric("Adoption rate", f"{adoption_rate:.1%}")
    metric_columns[2].metric("Median stay", f"{frame['length_of_stay'].median():.1f} days")

    st.subheader("Outcome mix")
    st.bar_chart(
        frame["Outcome Group"].value_counts().reindex(OUTCOME_ORDER, fill_value=0),
        color="#1565c0",
    )
    st.subheader("Adoption rate by intake condition")
    condition_table = (
        frame.groupby("Intake Condition")
        .agg(records=("Outcome Group", "size"), adoption_rate=("Outcome Group", lambda s: s.eq("Adoption").mean()),
             median_stay=("length_of_stay", "median"))
        .query("records >= 50")
        .sort_values("adoption_rate", ascending=False)
    )
    st.dataframe(
        condition_table.style.format(
            {"adoption_rate": "{:.1%}", "median_stay": "{:.1f}"}
        ),
        use_container_width=True,
    )
    st.subheader("Monthly completed outcomes")
    monthly = (
        frame.assign(month=frame["outcome_date"].dt.to_period("M").dt.to_timestamp())
        .pivot_table(index="month", columns="Outcome Group", values="Animal ID", aggfunc="count", fill_value=0)
        .reindex(columns=OUTCOME_ORDER, fill_value=0)
    )
    st.line_chart(monthly)


def render_forecast(intakes: pd.DataFrame, outcomes: pd.DataFrame, joined: pd.DataFrame) -> None:
    st.header("Forecast")
    st.caption(
        "A planning estimate based on the current active population and recent "
        "historical outcome proportions. Uncertainty is a bootstrap interval, "
        "not a full Bayesian forecast."
    )
    latest_intake = intakes.sort_values("intake_date").dropna(subset=["Animal ID"]).groupby("Animal ID").tail(1)
    latest_outcome = outcomes.sort_values("outcome_date").dropna(subset=["Animal ID"]).groupby("Animal ID").tail(1)
    current = latest_intake.merge(
        latest_outcome[["Animal ID", "outcome_date"]], on="Animal ID", how="left"
    )
    current = current[current["outcome_date"].isna() | (current["intake_date"] > current["outcome_date"])]
    recent = joined[joined["outcome_date"] >= joined["outcome_date"].max() - pd.Timedelta(days=730)]
    probabilities = _probabilities(recent)
    horizon = st.slider("Planning horizon (days)", 7, 90, 30, step=7)
    expected_intakes = st.number_input(
        "Expected new intakes during horizon", min_value=0, max_value=5000,
        value=max(1, round(len(current) * horizon / 365)), step=1
    )
    expected_counts = probabilities * expected_intakes
    metric_columns = st.columns(4)
    metric_columns[0].metric("Active population", f"{len(current):,}")
    metric_columns[1].metric("Expected intakes", f"{expected_intakes:,}")
    metric_columns[2].metric("Recent outcome records", f"{len(recent):,}")
    metric_columns[3].metric("Adoptions expected", f"{expected_counts['Adoption']:.0f}")

    rng = np.random.default_rng(42)
    draws = rng.multinomial(expected_intakes, probabilities.to_numpy(), size=2000)
    forecast = pd.DataFrame(draws, columns=OUTCOME_ORDER)
    summary = pd.DataFrame({
        "Expected": expected_counts,
        "Low (5%)": forecast.quantile(0.05),
        "High (95%)": forecast.quantile(0.95),
    })
    st.subheader("Expected outcomes")
    st.dataframe(summary.style.format("{:.0f}"), use_container_width=True)
    st.bar_chart(summary[["Expected"]], color="#2e7d32")
    st.caption(
        "Intervals capture outcome-mix uncertainty conditional on the intake "
        "count you entered; they do not account for seasonality or changing capacity."
    )


def main() -> None:
    st.set_page_config(page_title="Shelter decision support", page_icon="🐾", layout="wide")
    st.title("🐾 Shelter decision support")
    st.caption("Austin Animal Center · outcome, stay-length, and capacity planning")
    try:
        intakes, outcomes, joined = load_data()
    except (FileNotFoundError, pd.errors.ParserError) as error:
        st.error(f"Dashboard data could not be loaded: {error}")
        st.stop()

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["Predict", "Insights", "Forecast"])
        st.divider()
        st.caption("Data coverage")
        st.write(f"{intakes['intake_date'].min():%b %Y} – {intakes['intake_date'].max():%b %Y}")
        st.caption("Model status")
        st.info("Historical baseline active")

    if page == "Predict":
        render_predict(intakes, joined)
    elif page == "Insights":
        render_insights(intakes, joined)
    else:
        render_forecast(intakes, outcomes, joined)


if __name__ == "__main__":
    main()
