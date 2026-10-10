
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path
import math

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Food Waste Intelligence",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. PROFESSIONAL THEME AND READABLE TEXT
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f4f7f5;
        color: #24352b;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Main banner */
    .hero {
        background: linear-gradient(120deg, #12372a, #21845a);
        padding: 32px;
        border-radius: 18px;
        margin-bottom: 24px;
        color: #ffffff !important;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 32px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #edf8f0 !important;
        font-size: 16px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    /* Headings and general text */
    .section-heading {
        color: #183b2b !important;
        font-size: 24px;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 8px;
    }

    .explanation {
        color: #52665a !important;
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 16px;
    }

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown label {
        color: #26382e;
    }

    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4 {
        color: #183b2b;
    }

    /* Insight and recommendation cards */
    .insight-card {
        background-color: #e8f5ec !important;
        border-left: 5px solid #258653;
        border-radius: 10px;
        padding: 17px 19px;
        margin: 12px 0;
        color: #214b32 !important;
        line-height: 1.7;
    }

    .insight-card * {
        color: #214b32 !important;
    }

    .warning-card {
        background-color: #fff3d8 !important;
        border-left: 5px solid #d79a21;
        border-radius: 10px;
        padding: 17px 19px;
        margin: 12px 0;
        color: #674b13 !important;
        line-height: 1.7;
    }

    .warning-card * {
        color: #674b13 !important;
    }

    .metric-note {
        color: #52665a !important;
        font-size: 13px;
        line-height: 1.6;
    }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dfe9e2;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(30, 60, 40, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #52665a !important;
    }

    div[data-testid="stMetricValue"] {
        color: #176341 !important;
    }

    /* Upload section */
    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #dfe9e2;
        border-radius: 12px;
        padding: 14px;
    }

    /* Buttons */
    .stButton button,
    .stDownloadButton button {
        border-radius: 9px;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #edf3ee;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #617268 !important;
        padding: 20px 0;
        font-size: 13px;
        line-height: 1.8;
    }

    hr {
        border-color: #dce6df;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. MODEL LOCATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "food_waste_model.pkl"


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# =========================================================
# 4. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🍎 Food Waste AI")
    st.caption("Inventory Intelligence Dashboard")
    st.divider()

    st.markdown("### How to use this app")
    st.markdown("""
    1. Upload the preprocessed CSV.
    2. Review the input data.
    3. Generate predictions.
    4. Explore the charts.
    5. Review recommendations.
    6. Download the results.
    """)

    st.divider()
    st.markdown("### Built with")
    st.caption(
        "Python · Pandas · Scikit-learn · Plotly · Streamlit"
    )

    st.divider()
    st.caption(
        "Model predictions are estimates. Verify them against actual "
        "inventory and waste records before making decisions."
    )


# =========================================================
# 5. HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🍎 AI-Powered Food Waste Prediction System</h1>
    <p>
        Turn inventory data into understandable predictions.
        Explore potential waste, identify records for review,
        and make more informed inventory-management decisions.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 6. LOAD MODEL
# =========================================================
if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please check that "
        "'models/food_waste_model.pkl' exists in your GitHub repository."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.exception(exc)
    st.stop()


# =========================================================
# 7. UPLOAD DATASET
# =========================================================
st.markdown(
    '<div class="section-heading">📂 Upload Your Dataset</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'Upload the numeric, preprocessed feature dataset used by the trained '
    'model. Do not upload the raw dataset unless it has the same features '
    'and preprocessing as the training data.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose X_preprocessed.csv",
    type=["csv"],
    help="Upload the feature CSV that matches the model's training columns."
)

if uploaded_file is None:
    st.markdown("""
    <div class="insight-card">
        <b>Welcome!</b><br>
        Upload your preprocessed CSV file to begin. After prediction,
        the dashboard will show summary metrics, charts, high-prediction
        records, recommendations, and a downloadable results file.
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# =========================================================
# 8. READ AND VALIDATE INPUT
# =========================================================
try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read the uploaded CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()

if data.columns.duplicated().any():
    st.error("The uploaded CSV has duplicate column names.")
    st.stop()

if data.isna().any().any():
    st.error(
        "The uploaded CSV contains missing values. Please preprocess "
        "the data before uploading it."
    )
    st.stop()

non_numeric_columns = [
    col for col in data.columns
    if not pd.api.types.is_numeric_dtype(data[col])
]

if non_numeric_columns:
    st.error(
        "The uploaded dataset contains non-numeric columns: "
        + ", ".join(map(str, non_numeric_columns))
        + ". Upload the numeric, preprocessed feature dataset."
    )
    st.stop()

if not all(
    pd.Series(data[col]).map(
        lambda value: math.isfinite(float(value))
    ).all()
    for col in data.columns
):
    st.error(
        "The uploaded dataset contains infinite or invalid numeric values."
    )
    st.stop()


# =========================================================
# 9. DATASET OVERVIEW
# =========================================================
st.markdown(
    '<div class="section-heading">📋 Dataset Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

col1.metric("Input Records", f"{len(data):,}")
col2.metric("Input Features", f"{data.shape[1]:,}")
col3.metric("Missing Values", f"{int(data.isna().sum().sum()):,}")

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(10), use_container_width=True)


# =========================================================
# 10. PREDICTION
# =========================================================
st.markdown(
    '<div class="section-heading">🤖 Generate Predictions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'The trained Random Forest model estimates the waste quantity '
    'for each uploaded record.'
    '</div>',
    unsafe_allow_html=True
)

if st.button(
    "🚀 Predict Food Waste",
    type="primary",
    use_container_width=True
):
    try:
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
                        "Missing required features: "
                        + ", ".join(map(str, missing_features))
                    )

                if extra_features:
                    st.error(
                        "Unexpected columns: "
                        + ", ".join(map(str, extra_features))
                    )

                st.stop()

            # Use the exact training feature order.
            input_data = data[expected_features]
        else:
            input_data = data

        with st.spinner("Generating predictions..."):
            predictions = model.predict(input_data)

        prediction_series = pd.Series(predictions)

        if not prediction_series.map(
            lambda value: math.isfinite(float(value))
        ).all():
            st.error("The model returned invalid prediction values.")
            st.stop()

        result = data.copy()
        result["Predicted_Waste_Quantity"] = predictions

        st.session_state["food_waste_results"] = result

        st.success(
            f"Successfully generated predictions for {len(result):,} records."
        )

    except Exception as exc:
        st.error(
            "Prediction failed. Check that the uploaded columns and their "
            "preprocessing match the data used to train the model."
        )
        st.exception(exc)


# =========================================================
# 11. SHOW RESULTS
# =========================================================
if "food_waste_results" not in st.session_state:
    st.stop()

result = st.session_state["food_waste_results"].copy()
predicted = result["Predicted_Waste_Quantity"]

if result.empty:
    st.warning("No predictions are available.")
    st.stop()

total_waste = float(predicted.sum())
average_waste = float(predicted.mean())
maximum_waste = float(predicted.max())
minimum_waste = float(predicted.min())

highest_index = predicted.idxmax()
lowest_index = predicted.idxmin()

st.divider()

# =========================================================
# 12. SUMMARY METRICS
# =========================================================
st.markdown(
    '<div class="section-heading">📊 Prediction Summary</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'These figures summarize the predicted waste quantity across '
    'your uploaded records.'
    '</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

m1.metric("Total Predicted Waste", f"{total_waste:,.2f}")
m2.metric("Average per Record", f"{average_waste:,.2f}")
m3.metric("Highest Prediction", f"{maximum_waste:,.2f}")

st.markdown(
    '<p class="metric-note">'
    'The unit of measurement depends on the target variable used to train '
    'the model. These values are predictions, not verified measurements '
    'of actual waste.'
    '</p>',
    unsafe_allow_html=True
)


# =========================================================
# 13. EXPLAIN THE RESULTS IN PLAIN LANGUAGE
# =========================================================
st.markdown(
    '<div class="section-heading">💡 Understanding Your Results</div>',
    unsafe_allow_html=True
)

st.markdown(f"""
<div class="insight-card">
    <b>What does the total mean?</b><br>
    The model predicts a combined waste quantity of
    <b>{total_waste:,.2f}</b> across the {len(result):,} uploaded records.
    This is the sum of the model's estimates.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="insight-card">
    <b>What does the average mean?</b><br>
    The average predicted waste is <b>{average_waste:,.2f}</b>
    per record. Some records may have higher predictions and others lower.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="warning-card">
    <b>Which record should be reviewed first?</b><br>
    Input row <b>{int(highest_index) + 1}</b> has the highest predicted
    value of <b>{maximum_waste:,.2f}</b>. Check the corresponding inventory,
    demand, and expiry information before deciding what to do.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="insight-card">
    <b>What about the lowest prediction?</b><br>
    Input row <b>{int(lowest_index) + 1}</b> has the lowest predicted
    value of <b>{minimum_waste:,.2f}</b>. A low prediction does not
    guarantee that no waste will occur.
</div>
""", unsafe_allow_html=True)

st.info(
    "Input row numbers refer to the row positions in your uploaded CSV. "
    "They are not product IDs. The model's predictions alone do not "
    "identify the cause of potential waste."
)


# =========================================================
# 14. CHARTS
# =========================================================
st.markdown(
    '<div class="section-heading">📈 Explore the Predictions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'These charts help you compare predicted values and understand '
    'how they are distributed across records.'
    '</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2)

# Chart 1: Prediction distribution
with left:
    st.markdown("#### 1. Distribution of Predicted Waste")
    st.caption(
        "Each bar shows how many records fall within a range of predicted "
        "values. Taller bars indicate more records in that range."
    )

    fig_distribution = px.histogram(
        result,
        x="Predicted_Waste_Quantity",
        nbins=25,
        labels={
            "Predicted_Waste_Quantity": "Predicted waste quantity",
            "count": "Number of records"
        },
        color_discrete_sequence=["#258653"]
    )

    fig_distribution.update_layout(
        template="plotly_white",
        height=380,
        showlegend=False,
        margin=dict(l=15, r=15, t=25, b=15)
    )

    st.plotly_chart(
        fig_distribution,
        use_container_width=True
    )

# Chart 2: Top 10 predictions
with right:
    st.markdown("#### 2. Ten Highest Predictions")
    st.caption(
        "These are the records with the largest predicted waste quantities. "
        "Review their inventory details to determine whether action is needed."
    )

    top10 = result.nlargest(
        min(10, len(result)),
        "Predicted_Waste_Quantity"
    ).copy()

    top10["Input Record"] = [
        f"Row {int(index) + 1}" for index in top10.index
    ]

    top10 = top10.sort_values("Predicted_Waste_Quantity")

    fig_top10 = px.bar(
        top10,
        x="Predicted_Waste_Quantity",
        y="Input Record",
        orientation="h",
        labels={
            "Predicted_Waste_Quantity": "Predicted waste quantity",
            "Input Record": "Input record"
        },
        color_discrete_sequence=["#258653"]
    )

    fig_top10.update_layout(
        template="plotly_white",
        height=380,
        showlegend=False,
        margin=dict(l=15, r=15, t=25, b=15)
    )

    st.plotly_chart(
        fig_top10,
        use_container_width=True
    )

# Chart 3: Compare the first 30 input records
st.markdown("#### 3. Predicted Waste by Input Record")
st.caption(
    "Compares up to the first 30 records in uploaded order. "
    "The horizontal axis represents records, not dates or time."
)

record_chart = result[
    ["Predicted_Waste_Quantity"]
].head(30).copy()

record_chart["Input Record"] = [
    f"Row {i + 1}" for i in range(len(record_chart))
]

fig_records = px.line(
    record_chart,
    x="Input Record",
    y="Predicted_Waste_Quantity",
    markers=True,
    labels={
        "Input Record": "Input record",
        "Predicted_Waste_Quantity": "Predicted waste quantity"
    },
    color_discrete_sequence=["#258653"]
)

fig_records.update_layout(
    template="plotly_white",
    height=350,
    margin=dict(l=15, r=15, t=25, b=15)
)

st.plotly_chart(
    fig_records,
    use_container_width=True
)


# =========================================================
# 15. SUGGESTED NEXT ACTIONS
# =========================================================
st.markdown(
    '<div class="section-heading">🧭 Suggested Next Actions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'Use these general inventory-management suggestions as a starting point. '
    'They are not causes identified by the model.'
    '</div>',
    unsafe_allow_html=True
)

# HTML cards are used here to keep the text dark and readable.
# This prevents white text from blending into the light background.
st.markdown("""
<div class="insight-card">
    <b>1. Review high-prediction records</b><br>
    Check stock levels, recent sales, and expiry dates for records
    with high predicted waste. Confirm the details before taking action.
</div>

<div class="insight-card">
    <b>2. Check approaching expiry dates</b><br>
    Review products nearing expiry and consider safe stock rotation
    or other appropriate actions under food-safety rules.
</div>

<div class="insight-card">
    <b>3. Compare predictions with actual waste</b><br>
    Record actual waste quantities and compare them with the model's
    predictions to evaluate how well the model performs on real data.
</div>

<div class="insight-card">
    <b>4. Plan inventory carefully</b><br>
    Check verified sales and demand information before adjusting future
    stock orders. Do not change orders based only on a prediction.
</div>
""", unsafe_allow_html=True)


# =========================================================
# 16. DETAILED PREDICTION TABLE
# =========================================================
st.divider()

st.markdown(
    '<div class="section-heading">🧾 Detailed Prediction Results</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'Search through the input records and inspect their predicted waste '
    'quantities. The download includes all records, even when a filter '
    'is applied to this table.'
    '</div>',
    unsafe_allow_html=True
)

search_text = st.text_input(
    "Filter displayed records",
    placeholder="Enter a value to search in the table..."
)

display_result = result.copy()

if search_text.strip():
    matching_rows = result.astype(str).apply(
        lambda column: column.str.contains(
            search_text,
            case=False,
            regex=False
        )
    ).any(axis=1)

    display_result = result.loc[matching_rows]

st.caption(
    f"Showing {len(display_result):,} of {len(result):,} records."
)

st.dataframe(
    display_result,
    use_container_width=True,
    height=400
)


# =========================================================
# 17. DOWNLOAD PREDICTIONS
# =========================================================
st.markdown(
    '<div class="section-heading">⬇️ Download Your Results</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'Save the original uploaded feature values together with the '
    'model-generated predicted waste quantity.'
    '</div>',
    unsafe_allow_html=True
)

csv_data = result.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Prediction Results (CSV)",
    data=csv_data,
    file_name="food_waste_predictions.csv",
    mime="text/csv",
    type="primary",
    use_container_width=True
)


# =========================================================
# 18. MODEL LIMITATIONS
# =========================================================
with st.expander("ℹ️ About the predictions and their limitations"):
    st.markdown("""
    **How the system works**

    The application uses a trained Random Forest regression model to
    estimate waste quantity from the uploaded feature values.

    **How to interpret the results**

    - A high prediction identifies a record worth reviewing.
    - A prediction is an estimate, not a confirmed future outcome.
    - The charts describe model outputs, not verified real-world waste.
    - The model does not automatically identify why waste may occur.

    **Before making business decisions**

    Check the data quality, compare predictions with actual waste,
    and evaluate model performance on appropriate unseen data.
    """)


# =========================================================
# 19. FOOTER
# =========================================================
st.divider()

st.markdown("""
<div class="app-footer">
    <b>AI-Powered Food Waste Prediction System</b><br>
    Built with Python, Pandas, Scikit-learn, Plotly and Streamlit<br>
    <span>Predictions support inventory review; verify results before acting.</span>
</div>
""", unsafe_allow_html=True)
