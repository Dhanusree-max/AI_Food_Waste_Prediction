
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Food Waste Intelligence Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. PROFESSIONAL DESIGN
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f5f8f6;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #12372a, #21845a);
        padding: 30px;
        border-radius: 18px;
        margin-bottom: 24px;
        color: white;
    }

    .hero h1 {
        color: white;
        font-size: 32px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #e6f5ec;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-heading {
        color: #183b2b;
        font-size: 23px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 8px;
    }

    .explanation {
        color: #52665a;
        font-size: 14px;
        margin-bottom: 15px;
    }

    .insight {
        background: #e8f5ec;
        border-left: 5px solid #258653;
        padding: 16px;
        border-radius: 9px;
        margin: 10px 0;
        color: #214b32;
    }

    .warning {
        background: #fff4dc;
        border-left: 5px solid #d79a21;
        padding: 16px;
        border-radius: 9px;
        margin: 10px 0;
        color: #674b13;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e1eae4;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(30, 60, 40, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #5c7063;
    }

    div[data-testid="stMetricValue"] {
        color: #176341;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 12px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. MODEL PATH AND LOADING
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
    st.caption("Inventory Intelligence")
    st.divider()

    st.markdown("### How to use")
    st.write("1. Upload a preprocessed CSV.")
    st.write("2. Review the input data.")
    st.write("3. Generate predictions.")
    st.write("4. Understand the insights.")
    st.write("5. Download the results.")

    st.divider()

    st.markdown("### Technology")
    st.caption(
        "Python · Pandas · Scikit-learn · Plotly · Streamlit"
    )

    st.divider()
    st.caption(
        "Predictions are estimates from a trained machine learning model."
    )


# =========================================================
# 5. HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🍎 AI-Powered Food Waste Prediction System</h1>
    <p>
        Predict potential food waste, identify records that may need
        attention, and turn inventory data into understandable insights.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 6. LOAD MODEL
# =========================================================
if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Check that "
        "models/food_waste_model.pkl exists in your repository."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"The trained model could not be loaded: {exc}")
    st.stop()


# =========================================================
# 7. UPLOAD DATA
# =========================================================
st.markdown(
    '<div class="section-heading">📂 Upload Your Dataset</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'Upload the preprocessed feature dataset used by the trained model. '
    'The original raw dataset may not have the required columns.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose X_preprocessed.csv",
    type=["csv"],
    help="Use the same feature columns and data types used during training."
)

if uploaded_file is None:
    st.info(
        "👋 Welcome! Upload your preprocessed CSV file to start "
        "generating food waste predictions."
    )

    st.markdown("""
    <div class="insight">
        <b>What will you get?</b><br>
        A prediction summary, visual charts, a list of high-prediction
        records, understandable recommendations, and a downloadable report.
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# =========================================================
# 8. READ AND CHECK DATA
# =========================================================
try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()

if data.columns.duplicated().any():
    st.error("The CSV contains duplicate column names.")
    st.stop()

if data.isna().any().any():
    st.error(
        "The uploaded data contains missing values. "
        "Please preprocess the dataset before prediction."
    )
    st.stop()

if not all(
    pd.api.types.is_numeric_dtype(data[col])
    for col in data.columns
):
    st.error(
        "The uploaded feature dataset contains non-numeric columns. "
        "Please use the encoded, preprocessed feature dataset."
    )
    st.stop()

st.markdown(
    '<div class="section-heading">📋 Dataset Overview</div>',
    unsafe_allow_html=True
)

a, b, c = st.columns(3)
a.metric("Input Records", f"{len(data):,}")
b.metric("Input Features", f"{data.shape[1]:,}")
c.metric(
    "Missing Values",
    f"{int(data.isna().sum().sum()):,}"
)

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(10), use_container_width=True)


# =========================================================
# 9. GENERATE PREDICTIONS
# =========================================================
st.markdown(
    '<div class="section-heading">🤖 Generate Predictions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'Predictions are generated from the uploaded feature values '
    'using the saved Random Forest model.'
    '</div>',
    unsafe_allow_html=True
)

if st.button(
    "🚀 Predict Food Waste",
    type="primary",
    use_container_width=True
):
    try:
        # Align uploaded features to the model training schema
        expected_features = getattr(model, "feature_names_in_", None)

        if expected_features is not None:
            expected_features = list(expected_features)

            missing = [
                col for col in expected_features
                if col not in data.columns
            ]
            extra = [
                col for col in data.columns
                if col not in expected_features
            ]

            if missing or extra:
                if missing:
                    st.error(
                        "Required features missing: "
                        + ", ".join(missing)
                    )
                if extra:
                    st.error(
                        "Unexpected columns found: "
                        + ", ".join(extra)
                    )
                st.stop()

            input_data = data[expected_features]
        else:
            input_data = data

        with st.spinner("Generating predictions..."):
            predictions = model.predict(input_data)

        result = data.copy()
        result["Predicted_Waste_Quantity"] = predictions

        if not pd.Series(predictions).apply(
            lambda x: pd.notna(x) and abs(float(x)) != float("inf")
        ).all():
            st.error("The model produced invalid prediction values.")
            st.stop()

        # Save predictions across Streamlit reruns
        st.session_state["food_waste_results"] = result
        st.session_state["food_waste_input"] = input_data.copy()

        st.success(
            f"Predictions generated for {len(result):,} records."
        )

    except Exception as exc:
        st.error(
            "Prediction failed. Verify that the uploaded CSV contains "
            "the correct training features in compatible formats."
        )
        st.exception(exc)


# =========================================================
# 10. DISPLAY RESULTS
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

st.divider()

st.markdown(
    '<div class="section-heading">📊 Your Prediction Summary</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'These figures summarize the model predictions for your uploaded data.'
    '</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Total Predicted Waste",
    f"{total_waste:,.2f}",
    help="Sum of all predicted waste quantities in the uploaded records."
)

m2.metric(
    "Average Predicted Waste",
    f"{average_waste:,.2f}",
    help="Mean predicted waste quantity per record."
)

m3.metric(
    "Highest Predicted Waste",
    f"{maximum_waste:,.2f}",
    help="Largest predicted waste quantity among the uploaded records."
)

st.caption(
    "Units follow the target variable used to train the model. "
    "These are estimates, not verified measurements of actual waste."
)


# =========================================================
# 11. PLAIN-LANGUAGE INTERPRETATION
# =========================================================
st.markdown(
    '<div class="section-heading">💡 What Do These Numbers Mean?</div>',
    unsafe_allow_html=True
)

highest_index = predicted.idxmax()
highest_value = float(predicted.loc[highest_index])

average_index = predicted.idxmin()
lowest_value = float(predicted.loc[average_index])

st.markdown(f"""
<div class="insight">
    <b>Overall estimate</b><br>
    Across {len(result):,} uploaded records, the model predicts a combined
    waste quantity of <b>{total_waste:,.2f}</b> target units.
    The average prediction is <b>{average_waste:,.2f}</b> per record.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="warning">
    <b>Record needing the closest review</b><br>
    The record at input row <b>{int(highest_index) + 1}</b> has the highest
    predicted value: <b>{highest_value:,.2f}</b>. Review the corresponding
    inventory information to decide whether action is needed.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="insight">
    <b>Lowest prediction in this dataset</b><br>
    The record at input row <b>{int(average_index) + 1}</b> has the lowest
    predicted value: <b>{lowest_value:,.2f}</b>. A lower prediction does
    not mean there is zero risk of waste.
</div>
""", unsafe_allow_html=True)

st.info(
    "These comparisons describe model output only. They do not establish "
    "why a record has high or low waste, or whether a difference is "
    "statistically significant."
)


# =========================================================
# 12. INTERACTIVE CHARTS
# =========================================================
st.markdown(
    '<div class="section-heading">📈 Explore the Predictions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'Use these charts to understand how predicted values vary across '
    'your uploaded records.'
    '</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2)

# Chart 1: Distribution
with left:
    st.markdown("#### 1. Distribution of Predictions")
    st.caption(
        "Shows how many records fall into different predicted-waste ranges. "
        "Taller bars mean more records fall within that range."
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

# Chart 2: Highest predictions
with right:
    st.markdown("#### 2. Ten Highest Predictions")
    st.caption(
        "Highlights the records with the largest predicted quantities. "
        "These are useful candidates for inventory review."
    )

    top10 = (
        result.nlargest(10, "Predicted_Waste_Quantity")
        .copy()
    )

    # Display rows as user-friendly labels, avoiding false product names
    top10["Record"] = [
        f"Input row {i + 1}"
        for i in top10.index
    ]

    top10 = top10.sort_values("Predicted_Waste_Quantity")

    fig_top10 = px.bar(
        top10,
        x="Predicted_Waste_Quantity",
        y="Record",
        orientation="h",
        labels={
            "Predicted_Waste_Quantity": "Predicted waste quantity",
            "Record": "Input record"
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

# Chart 3: Record comparison
st.markdown("#### 3. Predicted Waste by Record")
st.caption(
    "Compares the first 30 records in their uploaded order. "
    "Each point is a prediction, not a measurement over time."
)

record_chart_data = result[
    ["Predicted_Waste_Quantity"]
].head(30).copy()

record_chart_data["Input Record"] = [
    f"Row {i + 1}"
    for i in range(len(record_chart_data))
]

fig_records = px.line(
    record_chart_data,
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

st.plotly_chart(fig_records, use_container_width=True)


# =========================================================
# 13. ACTIONABLE GUIDANCE
# =========================================================
st.markdown(
    '<div class="section-heading">🧭 Suggested Next Actions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="explanation">'
    'These are general inventory-management suggestions, not causes '
    'identified by the model.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
- **Review high-prediction records:** check stock levels, sales trends,
  product expiry dates, and recent demand.
- **Check upcoming expiry dates:** prioritize safe, eligible stock
  rotation and follow applicable food-safety rules.
- **Compare with actual outcomes:** track observed waste over time
  to evaluate how useful the predictions are.
- **Adjust inventory carefully:** use verified sales and demand data
  before changing order quantities.
""")


# =========================================================
# 14. RESULTS TABLE
# =========================================================
st.divider()

st.markdown(
    '<div class="section-heading">🧾 Detailed Prediction Results</div>',
    unsafe_allow_html=True
)

st.caption(
    "Input row numbers refer to the row order in the uploaded CSV. "
    "They are not product IDs."
)

search_text = st.text_input(
    "Filter displayed records",
    placeholder="Search values in the prediction table..."
)

display_result = result.copy()

if search_text.strip():
    matches = result.astype(str).apply(
        lambda column: column.str.contains(
            search_text,
            case=False,
            regex=False
        )
    ).any(axis=1)

    display_result = result[matches]

st.caption(
    f"Displaying {len(display_result):,} of {len(result):,} records."
)

st.dataframe(
    display_result,
    use_container_width=True,
    height=400
)


# =========================================================
# 15. DOWNLOAD RESULTS
# =========================================================
st.markdown(
    '<div class="section-heading">⬇️ Download Your Report</div>',
    unsafe_allow_html=True
)

st.write(
    "Download the input features together with the predicted waste "
    "quantity for further analysis."
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


# =========================================================
# 16. FOOTER AND LIMITATIONS
# =========================================================
st.divider()

with st.expander("About these predictions and their limitations"):
    st.write(
        "This application uses a trained Random Forest regression model "
        "to estimate waste quantity from the uploaded features."
    )
    st.write(
        "Prediction quality depends on the training data, input quality, "
        "and whether new records resemble the data used during training."
    )
    st.write(
        "High predicted values indicate records worth reviewing; they do "
        "not prove that waste will occur or identify its cause."
    )
    st.write(
        "Validate predictions against actual observed waste before using "
        "them for operational or financial decisions."
    )

st.markdown("""
<div style="text-align:center; color:#708077; padding:18px;">
    <b>AI-Powered Food Waste Prediction System</b><br>
    Python · Scikit-learn · Pandas · Plotly · Streamlit
</div>
""", unsafe_allow_html=True)
