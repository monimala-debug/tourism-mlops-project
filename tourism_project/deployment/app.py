"""
app.py - Streamlit front end for the tourism purchase-propensity model.

Streamlit Community Cloud runs this file directly from the GitHub repository
(main file path: tourism_project/deployment/app.py). It loads model.joblib, which the
GitHub Actions workflow trains and commits next to this file. Until that first workflow
run has finished, the app shows an explanatory error instead of a prediction.
"""

import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# The model sits next to this file. TOURISM_MODEL_PATH lets the Colab notebook smoke-test
# the app against a locally trained model without touching the committed file.
MODEL_PATH = Path(os.environ.get(
    "TOURISM_MODEL_PATH", Path(__file__).resolve().parent / "model.joblib"
))
DECISION_THRESHOLD = 0.5  # must match DECISION_THRESHOLD in train.py

# Exactly the features the model was trained on (same names, same order as prep.py).
NUMERIC_FEATURES = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = ["Occupation", "Gender", "MaritalStatus", "Designation"]
EXPECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Cleaned category labels (after prep.py's clean-up) offered in the drop-downs.
OCCUPATIONS = ["Salaried", "Small Business", "Large Business", "Freelancer"]
GENDERS = ["Male", "Female"]
MARITAL_STATUSES = ["Married", "Single", "Unmarried", "Divorced"]
DESIGNATIONS = ["Executive", "Manager", "Senior Manager", "AVP", "VP"]


def build_input_frame(values: dict) -> pd.DataFrame:
    """Turn raw form values into a one-row DataFrame with the training column names,
    order and types (numbers as float64, categories as text)."""
    missing = [f for f in EXPECTED_FEATURES if f not in values]
    extra = [k for k in values if k not in EXPECTED_FEATURES]
    if missing or extra:
        raise ValueError(f"Input mismatch. Missing: {missing}; unexpected: {extra}")
    row = pd.DataFrame([{f: values[f] for f in EXPECTED_FEATURES}])
    row[NUMERIC_FEATURES] = row[NUMERIC_FEATURES].astype("float64")
    row[CATEGORICAL_FEATURES] = row[CATEGORICAL_FEATURES].astype(str)
    return row


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def main() -> None:
    st.set_page_config(page_title="Tourism Package Propensity", page_icon=":airplane:")
    st.title("Tourism package purchase propensity")
    st.write(
        "Estimate how likely a customer is to buy a tourism package, using only "
        "information available **before** a salesperson contacts them."
    )
    st.info(
        "The model was trained on historical purchases of the company's existing tourism "
        "packages. It has not been validated on Wellness Tourism Package sales, so treat "
        "the score as a prioritisation aid, not a guarantee."
    )

    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at `{MODEL_PATH}`. Run the GitHub Actions pipeline "
            "once: it trains the model and commits `model.joblib` to "
            "`tourism_project/deployment/`. Then reboot this app."
        )
        st.stop()

    model = load_model(str(MODEL_PATH))
    trained_features = list(getattr(model, "feature_names_in_", EXPECTED_FEATURES))
    if sorted(trained_features) != sorted(EXPECTED_FEATURES):
        st.error(f"The model expects {trained_features}, but this form collects "
                 f"{EXPECTED_FEATURES}. Retrain the model or update the app.")
        st.stop()

    with st.form("customer_form"):
        st.subheader("Customer profile")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=36, step=1)
            gender = st.selectbox("Gender", GENDERS)
            marital = st.selectbox("Marital status", MARITAL_STATUSES)
            occupation = st.selectbox("Occupation", OCCUPATIONS)
            designation = st.selectbox("Designation", DESIGNATIONS)
            income = st.number_input("Gross monthly income", min_value=0.0,
                                     value=22000.0, step=500.0)
            city_tier = st.selectbox("City tier (1 = most developed)", [1, 2, 3])
        with c2:
            persons = st.number_input("Number of people on the trip", 1, 10, 3, 1)
            children = st.number_input("Children under 5 on the trip", 0, 5, 1, 1)
            trips = st.number_input("Average trips per year", 0, 30, 3, 1)
            star = st.selectbox("Preferred hotel star rating", [3, 4, 5])
            passport = st.radio("Has a valid passport?", ["No", "Yes"], horizontal=True)
            own_car = st.radio("Owns a car?", ["No", "Yes"], horizontal=True)
        submitted = st.form_submit_button("Estimate purchase likelihood")

    if submitted:
        values = {
            "Age": age, "CityTier": city_tier, "NumberOfPersonVisiting": persons,
            "PreferredPropertyStar": star, "NumberOfTrips": trips,
            "Passport": 1 if passport == "Yes" else 0,
            "OwnCar": 1 if own_car == "Yes" else 0,
            "NumberOfChildrenVisiting": children, "MonthlyIncome": income,
            "Occupation": occupation, "Gender": gender,
            "MaritalStatus": marital, "Designation": designation,
        }
        row = build_input_frame(values)[trained_features]
        probability = float(model.predict_proba(row)[0, 1])
        likely = probability >= DECISION_THRESHOLD

        st.subheader("Estimate")
        st.metric("Estimated purchase probability", f"{probability:.1%}")
        if likely:
            st.success(f"Above the {DECISION_THRESHOLD:.0%} cut-off: a priority prospect.")
        else:
            st.warning(f"Below the {DECISION_THRESHOLD:.0%} cut-off: lower priority.")
        st.caption(
            "This is a statistical estimate from historical data, not a prediction of "
            "certainty. Customers far outside the training data (for example age above "
            "61 or unusual income) get less reliable scores."
        )
        with st.expander("Model input sent for this estimate"):
            st.dataframe(row)


if __name__ == "__main__":
    # Streamlit runs this file as __main__; importing it (e.g. in tests) does not
    # start the user interface.
    main()
