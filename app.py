
import os
import joblib
import pandas as pd
import streamlit as st
import gzip

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="California Housing | Value Estimator",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# 2. PROFESSIONAL CSS THEME
# =========================================================

st.markdown("""
<style>

/* Overall page */
.stApp {
    background: #F5F6F2;
    color: #26332D;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Typography */
h1, h2, h3 {
    color: #244C3A !important;
    font-weight: 750 !important;
}

p, label, li {
    color: #46534B;
}

[data-testid="stCaptionContainer"] {
    color: #727D75;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #183D30, #35664D);
    padding: 36px 40px;
    border-radius: 18px;
    margin-bottom: 25px;
    color: white;
}

.hero h1 {
    color: white !important;
    font-size: 2.35rem;
    margin-bottom: 8px;
}

.hero p {
    color: #E1EDE4;
    font-size: 1.05rem;
    margin-bottom: 0;
    line-height: 1.7;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    color: #F2F7F2;
    border: 1px solid rgba(255,255,255,0.25);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    margin-bottom: 14px;
}

/* Section cards */
.section-card {
    background: #FFFFFF;
    border: 1px solid #E0E6DE;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 18px;
}

/* Input fields */
.stNumberInput input {
    background: #FFFFFF !important;
    color: #26332D !important;
    border: 1px solid #CBD5CC !important;
    border-radius: 9px !important;
}

.stNumberInput label {
    color: #315A43 !important;
    font-weight: 650 !important;
}

/* Input focus */
.stNumberInput input:focus {
    border-color: #477A59 !important;
    box-shadow: 0 0 0 1px #477A59 !important;
}

/* Buttons */
.stFormSubmitButton > button,
.stButton > button {
    background: #D97848 !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 0.7rem 1.2rem !important;
    font-weight: 700 !important;
    transition: 0.2s ease;
}

.stFormSubmitButton > button:hover,
.stButton > button:hover {
    background: #B85E35 !important;
    color: white !important;
}

/* Result card */
.result-card {
    background: #FFFFFF;
    border: 1px solid #DCE6DC;
    border-left: 5px solid #3E7954;
    border-radius: 14px;
    padding: 25px;
    margin-top: 15px;
    margin-bottom: 15px;
}

.result-label {
    color: #65746A;
    font-size: 0.9rem;
    font-weight: 600;
}

.result-value {
    color: #24583D;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 5px 0 12px 0;
}

.interval-value {
    color: #315C45;
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 5px;
}

/* Information panels */
.info-panel {
    background: #EAF0E9;
    border: 1px solid #D3E0D3;
    border-radius: 12px;
    padding: 17px 20px;
    margin-top: 14px;
    margin-bottom: 14px;
}

.info-panel p {
    margin-bottom: 5px;
    line-height: 1.6;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E0E6DE;
    border-radius: 12px;
    padding: 16px;
}

div[data-testid="stMetricLabel"] {
    color: #65746A !important;
}

div[data-testid="stMetricValue"] {
    color: #28563D !important;
    font-weight: 750 !important;
}

/* Divider */
hr {
    border-color: #DCE3DB;
}

/* Footer */
.footer {
    text-align: center;
    color: #78847B;
    font-size: 0.85rem;
    padding-top: 20px;
    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD MODEL AND CALIBRATION THRESHOLD
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "california_housing_model.pkl.gz"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR, "calibration_threshold.pkl.gz"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_threshold():
    return joblib.load(THRESHOLD_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        "The trained model file was not found. "
        "Place california_housing_model.pkl in the same folder as this app."
    )
    st.stop()

if not os.path.exists(THRESHOLD_PATH):
    st.error(
        "The calibration threshold file was not found. "
        "Place calibration_threshold.pkl in the same folder as this app."
    )
    st.stop()

try:
    model = load_model()
    calibration_threshold = float(load_threshold())
except Exception as error:
    st.error(f"Unable to load the model files: {error}")
    st.stop()


# =========================================================
# 4. WEBSITE HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">MACHINE LEARNING PROJECT · CALIFORNIA, USA</div>
    <h1>California Housing Value Estimator</h1>
    <p>
        An interactive machine learning application that estimates
        median housing values across California using demographic,
        housing and geographical information.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 5. PROJECT INTRODUCTION
# =========================================================

st.markdown("## Project Overview")

st.markdown("""
<div class="section-card">
    <p>
        This application uses a <b>Random Forest Regression model</b>
        trained on the California Housing dataset, derived from the
        <b>1990 U.S. Census</b>.
    </p>
    <p>
        Enter eight numerical features describing a California
        geographical area. The model processes these inputs and
        estimates the area's median house value.
    </p>
    <p>
        The application also provides a <b>90% prediction interval</b>
        to communicate uncertainty around the estimate.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================

# =========================================================
# 6. HOUSING INPUT FORM
# =========================================================

st.markdown("## Enter Housing and Area Details")

st.caption(
    "Enter the characteristics of a California geographical area. "
    "Use the plus and minus controls to adjust the values."
)

with st.form("housing_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Housing Characteristics")

        med_inc = st.number_input(
            "Median Income (units of $10,000)",
            min_value=0.50,
            max_value=15.00,
            value=5.00,
            step=0.10,
            format="%.2f",
            help="5.00 represents approximately $50,000 in median area income."
        )

        house_age = st.number_input(
            "Median House Age (years)",
            min_value=1,
            max_value=52,
            value=20,
            step=1,
            format="%d",
            help="Median age of houses in the area."
        )

        ave_rooms = st.number_input(
            "Average Rooms per Household",
            min_value=0.85,
            max_value=141.91,
            value=5.00,
            step=0.10,
            format="%.2f",
            help="Average number of rooms per household."
        )

        ave_bedrms = st.number_input(
            "Average Bedrooms per Household",
            min_value=0.33,
            max_value=34.07,
            value=1.00,
            step=0.10,
            format="%.2f",
            help="Average number of bedrooms per household."
        )

    with col2:

        st.markdown("### Population and Location")

        population = st.number_input(
            "Area Population",
            min_value=3,
            max_value=35682,
            value=1000,
            step=1,
            format="%d",
            help="Total population of the geographical area."
        )

        ave_occup = st.number_input(
            "Average Household Occupancy",
            min_value=0.69,
            max_value=1243.34,
            value=3.00,
            step=0.10,
            format="%.2f",
            help="Average number of people per household."
        )

        latitude = st.number_input(
            "Latitude",
            min_value=32.54,
            max_value=41.95,
            value=34.05,
            step=0.01,
            format="%.2f",
            help="Geographical latitude of the area."
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-124.35,
            max_value=-114.31,
            value=-118.25,
            step=0.01,
            format="%.2f",
            help="Geographical longitude of the area."
        )

    st.markdown("---")

    submit_button = st.form_submit_button(
        "Generate Housing Value Estimate",
        use_container_width=True
    )



# =========================================================
# 7. PREDICTION AND UNCERTAINTY INTERVAL
# =========================================================

if submit_button:

    # Keep the feature order identical to model training.
    input_data = pd.DataFrame(
        [[
            med_inc,
            house_age,
            ave_rooms,
            ave_bedrms,
            population,
            ave_occup,
            latitude,
            longitude
        ]],
        columns=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]
    )

    try:

        prediction = float(model.predict(input_data)[0])

        # Dataset target is measured in units of $100,000.
        estimated_value = prediction * 100000

        # Split-conformal prediction interval.
        lower_bound = max(
            0.0,
            prediction - calibration_threshold
        )

        upper_bound = (
            prediction + calibration_threshold
        )

        lower_dollars = lower_bound * 100000
        upper_dollars = upper_bound * 100000

        interval_width = (
            upper_dollars - lower_dollars
        )

        st.divider()
        st.markdown("## Prediction Results")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">
                ESTIMATED MEDIAN AREA-LEVEL HOUSE VALUE
            </div>
            <div class="result-value">
                ${estimated_value:,.2f}
            </div>
            <div class="result-label">
                Random Forest Regression Estimate
            </div>
        </div>
        """, unsafe_allow_html=True)

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                label="Lower Prediction Bound",
                value=f"${lower_dollars:,.2f}"
            )

        with result_col2:
            st.metric(
                label="Upper Prediction Bound",
                value=f"${upper_dollars:,.2f}"
            )

        st.markdown(f"""
        <div class="info-panel">
            <p><b>90% Prediction Interval</b></p>
            <p>
                The estimated value is expected to fall within the
                displayed interval under the calibration method's
                assumptions.
            </p>
            <p>
                <b>Interval width:</b> ${interval_width:,.2f}
            </p>
            <p>
                <b>Calibration margin:</b>
                ±${calibration_threshold * 100000:,.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Understanding the Prediction")

        st.write(
            "The Random Forest model estimates the median housing "
            "value using the eight characteristics entered above."
        )

        st.write(
            "The prediction interval is calculated using split "
            "conformal calibration. The calibration threshold is "
            "based on the absolute prediction errors observed on "
            "a separate calibration dataset."
        )

        st.caption(
            "The 90% figure is the target coverage level of the "
            "prediction interval. It is not a probability that "
            "this individual prediction is correct."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# =========================================================
# 8. MODEL RELIABILITY INFORMATION
# =========================================================

st.divider()
st.markdown("## Model Reliability and Methodology")

reliability_col1, reliability_col2, reliability_col3 = st.columns(3)

with reliability_col1:
    st.metric(
        label="Target Interval Coverage",
        value="90%"
    )

with reliability_col2:
    st.metric(
        label="Observed Test Coverage",
        value="90.12%"
    )

with reliability_col3:
    st.metric(
        label="Calibration Samples",
        value="3,303"
    )

st.markdown("""
<div class="section-card">
    <p><b>How is the interval determined?</b></p>
    <p>
        The model first predicts values for the calibration dataset.
        The absolute differences between actual and predicted values
        are calculated. The 90th-percentile calibration error is then
        used as the margin around a new prediction.
    </p>
    <p>
        The resulting interval is evaluated on a separate test dataset.
        In this project, 90.12% of the held-out test observations fell
        within their prediction intervals.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 9. ABOUT THE DATASET AND LIMITATIONS
# =========================================================

st.divider()
st.markdown("## About the Dataset")

with st.expander("Dataset, features and limitations", expanded=False):

    st.markdown("""
    **Dataset source:** California Housing dataset  
    **Data period:** 1990 U.S. Census  
    **Total observations:** 20,640  
    **Input features:** 8 numerical variables  
    **Prediction target:** Median house value of a geographical area  

    ### Model inputs

    | Feature | Description |
    |---|---|
    | MedInc | Median income of the area |
    | HouseAge | Median house age |
    | AveRooms | Average rooms per household |
    | AveBedrms | Average bedrooms per household |
    | Population | Area population |
    | AveOccup | Average household occupancy |
    | Latitude | Geographical latitude |
    | Longitude | Geographical longitude |

    ### Important limitations

    - The output represents an area-level median value, not the
      selling price of an individual property.
    - The dataset reflects historical 1990 conditions and should
      not be treated as a current real-estate valuation.
    - The original target values are capped at $500,000.
    - Prediction intervals rely on the assumption that future
      examples are sufficiently similar to the calibration data.
    - Changes in location, economic conditions and housing markets
      can affect prediction reliability.
    """)


# =========================================================
# 10. FOOTER
# =========================================================

st.markdown("""
<hr>
<div class="footer">
    <b>California Housing Value Estimator</b><br>
    Developed using Python · Pandas · Scikit-learn · Streamlit<br>
    Machine Learning Regression Project
</div>
""", unsafe_allow_html=True)
