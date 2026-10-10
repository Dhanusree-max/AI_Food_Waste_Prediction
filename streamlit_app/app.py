
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Food Waste Intelligence",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PROFESSIONAL STYLING
# =====================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        background: linear-gradient(120deg, #12372a, #227653);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 32px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e1f4e8;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-title {
        color: #183b2b;
        font-size: 22px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e3e9e5;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(20, 50, 35, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #607166;
    }

    div[data-testid="stMetricValue"] {
        color: #176341;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        padding: 16px;
        border: 1px dashed #78a88d;
        border-radius: 12px;
    }

    .note {
        background: #eaf5ee;
        border-left: 4px solid #238451;
        padding: 12px 16px;
        border-radius: 6px;
        color: #244b35;
        margin: 12px 0 20px 0;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# MODEL LOADING
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "food_waste_model.pkl"


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 🍎 Food Waste AI")
    st.caption("Intelligent Inventory Analytics")
    st.divider()

    st.markdown("### How it works")
    st.write("1. Upload the preprocessed feature CSV.")
    st.write("2. Review the uploaded data.")
    st.write("3. Generate waste predictions.")
    st.write("4. Explore charts and download results.")

    st.divider()
    st.markdown("**Technology stack**")
    st.caption("Python · Scikit-learn · Pandas · Plotly · Streamlit")


# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class="hero">
    <h1>🍎 AI-Powered Food Waste Prediction</h1>
    <p>
        Turn inventory data into actionable insights.
        Predict potential food waste and explore patterns
        through an interactive analytics dashboard.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL SAFELY
# =====================================================
if not MODEL_PATH.exists():
    st.error(
        "The trained model file could not be found. "
        "Please check that models/food_waste_model.pkl "
        "exists in the GitHub repository."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"The trained model could not be loaded: {exc}")
    st.stop()


# =====================================================
# DATA UPLOAD
# =====================================================
st.markdown(
    '<div class="section-title">📂 Upload Your Dataset</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload the `X_preprocessed.csv` file containing the "
    "features used during model training."
)

uploaded_file = st.file_uploader(
    "Choose a preprocessed CSV file",
    type=["csv"],
    help="Upload a CSV with the same feature columns used to train the model."
)

if uploaded_file is None:
    st.info(
        "👋 Welcome! Upload your preprocessed CSV file to begin "
        "generating food waste predictions."
    )

    st.markdown("""
    <div class="note">
        <b>Before you begin:</b> This application expects the same
        input features used to train the saved machine learning model.
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# =====================================================
# READ AND VALIDATE DATA
# =====================================================
try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV file: {exc}")
    st.stop()

if data.empty:
    st.warning("The uploaded CSV contains no records.")
    st.stop()

if data.columns.duplicated().any():
    st.error("The CSV contains duplicate column names. Please correct it.")
    st.stop()

st.markdown(
    '<div class="section-title">📋 Dataset Overview</div>',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)
info1.metric("Total Records", f"{len(data):,}")
info2.metric("Input Features", f"{data.shape[1]:,}")
info3.metric("Missing Values", f"{int(data.isna().sum().sum()):,}")

with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(data.head(10), use_container_width=True)


# =====================================================
# PREDICTION BUTTON
# =====================================================
st.markdown(
    '<div class="section-title">🤖 Generate Predictions</div>',
    unsafe_allow_html=True
)

if st.button(
    "🚀 Predict Food Waste",
    type="primary",
    use_container_width=True
):
    try:
        # Check missing values
        if data.isna().any().any():
            st.error(
                "The uploaded dataset contains missing values. "
                "Please preprocess the data before prediction."
            )
            st.stop()

        # Check the trained model's expected feature names
        expected_features = getattr(model, "feature_names_in_", None)

        if expected_features is not None:
            expected_features = list(expected_features)
            missing_features = [
                col for col in expected_features if col not in data.columns
            ]
            extra_features = [
                col for col in data.columns if col not in expected_features
            ]

            if missing_features or extra_features:
                if missing_features:
                    st.error(
                        "Missing required model features: "
                        + ", ".join(missing_features)
                    )
                if extra_features:
                    st.error(
                        "Unexpected input columns: "
                        + ", ".join(extra_features)
                    )
                st.stop()

            # Preserve the model's training feature order
            data = data[expected_features]

        # Generate predictions
        with st.spinner("Analyzing the dataset and generating predictions..."):
            predictions = model.predict(data)

        result = data.copy()
        result["Predicted_Waste_Quantity"] = predictions

        st.success("Predictions generated successfully!")

        # Keep results for this script run
        st.session_state["food_waste_results"] = result

    except Exception as exc:
        st.error(
            "Prediction failed. Please verify that the uploaded CSV "
            "contains the correct preprocessed features."
        )
        st.exception(exc)


# =====================================================
# DISPLAY PREDICTIONS
# =====================================================
if "food_waste_results" in st.session_state:
    result = st.session_state["food_waste_results"]

    predicted = result["Predicted_Waste_Quantity"]

    st.divider()
    st.markdown(
        '<div class="section-title">📈 Prediction Summary</div>',
        unsafe_allow_html=True
    )

    total_waste = predicted.sum()
    average_waste = predicted.mean()
    maximum_waste = predicted.max()

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Total Predicted Waste",
        f"{total_waste:,.2f}"
    )
    metric2.metric(
        "Average Predicted Waste",
        f"{average_waste:,.2f}"
    )
    metric3.metric(
        "Maximum Predicted Waste",
        f"{maximum_waste:,.2f}"
    )

    st.caption(
        "These figures summarize model predictions for the uploaded records. "
        "They are not verified measurements of actual food waste."
    )

    # =================================================
    # INTERACTIVE CHARTS
    # =================================================
    st.markdown(
        '<div class="section-title">📊 Interactive Analytics</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    # Chart 1: Distribution
    with left:
        st.markdown("#### Predicted Waste Distribution")

        histogram = px.histogram(
            result,
            x="Predicted_Waste_Quantity",
            nbins=25,
            color_discrete_sequence=["#21865b"],
            labels={
                "Predicted_Waste_Quantity": "Predicted Waste Quantity",
                "count": "Number of Records"
            },
            title="Distribution Across Records"
        )

        histogram.update_layout(
            template="plotly_white",
            height=380,
            showlegend=False,
            margin=dict(l=15, r=15, t=55, b=15)
        )

        st.plotly_chart(histogram, use_container_width=True)

    # Chart 2: Top 10
    with right:
        st.markdown("#### Top 10 Highest Predictions")

        top10 = (
            result.nlargest(10, "Predicted_Waste_Quantity")
            .copy()
        )

        top10["Record"] = [
            f"Record {i}" for i in range(1, len(top10) + 1)
        ]

        top10 = top10.sort_values("Predicted_Waste_Quantity")

        bar_chart = px.bar(
            top10,
            x="Predicted_Waste_Quantity",
            y="Record",
            orientation="h",
            color="Predicted_Waste_Quantity",
            color_continuous_scale=["#a8dfbd", "#176341"],
            labels={
                "Predicted_Waste_Quantity": "Predicted Waste",
                "Record": "Record"
            },
            title="Records With Highest Predicted Waste"
        )

        bar_chart.update_layout(
            template="plotly_white",
            height=380,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=15, r=15, t=55, b=15)
        )

        st.plotly_chart(bar_chart, use_container_width=True)

    # Chart 3: First 30 records
    st.markdown("#### Predicted Waste by Record")

    chart_data = result[
        ["Predicted_Waste_Quantity"]
    ].head(30).copy()

    chart_data["Record"] = [
        f"Record {i}" for i in range(1, len(chart_data) + 1)
    ]

    record_chart = px.line(
        chart_data,
        x="Record",
        y="Predicted_Waste_Quantity",
        markers=True,
        color_discrete_sequence=["#21865b"],
        labels={
            "Record": "Input Record",
            "Predicted_Waste_Quantity": "Predicted Waste"
        },
        title="Predictions for the First 30 Records"
    )

    record_chart.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=15, r=15, t=55, b=15)
    )

    st.plotly_chart(record_chart, use_container_width=True)

    # =================================================
    # RESULTS TABLE
    # =================================================
    st.divider()
    st.markdown(
        '<div class="section-title">🧾 Detailed Prediction Results</div>',
        unsafe_allow_html=True
    )

    search_text = st.text_input(
        "Filter results by text",
        placeholder="Enter a value to filter the displayed table..."
    )

    display_result = result

    if search_text.strip():
        mask = result.astype(str).apply(
            lambda column: column.str.contains(
                search_text,
                case=False,
                regex=False
            )
        ).any(axis=1)

        display_result = result[mask]

    st.caption(f"Showing {len(display_result):,} of {len(result):,} records.")

    st.dataframe(
        display_result,
        use_container_width=True,
        height=400
    )

    # =================================================
    # DOWNLOAD
    # =================================================
    st.markdown(
        '<div class="section-title">⬇️ Export Your Results</div>',
        unsafe_allow_html=True
    )

    csv_data = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Results (CSV)",
        data=csv_data,
        file_name="food_waste_predictions.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True
    )


# =====================================================
# FOOTER
# =====================================================
st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#718078; padding:12px;">
        <b>AI-Powered Food Waste Prediction System</b><br>
        Built with Python, Scikit-learn, Plotly and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
