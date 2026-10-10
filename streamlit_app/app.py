import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ============================================================
# Page configuration and professional theme
# ============================================================
st.set_page_config(
    page_title="Food Waste Intelligence",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #F5F7FA;
        color: #172B4D;
    }
    [data-testid="stHeader"] {
        background: rgba(245,247,250,0.96);
    }
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] * {
        color: #26364B;
    }
    .hero {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 5px solid #0F766E;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    }
    .hero h1 {
        color: #172B4D;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 10px 0;
        line-height: 1.25;
    }
    .hero p {
        color: #52647A;
        font-size: 1rem;
        margin: 0;
        line-height: 1.7;
    }
    .section-title {
        color: #172B4D;
        font-size: 1.35rem;
        font-weight: 750;
        margin: 20px 0 8px 0;
    }
    .section-help {
        color: #52647A;
        font-size: 0.93rem;
        margin-bottom: 14px;
        line-height: 1.6;
    }
    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 112px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
    }
    .info-label {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .info-value {
        color: #172B4D;
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .info-note {
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 7px;
    }
    .recommendation {
        background: #FFFFFF;
        border: 1px solid #DCE5EC;
        border-left: 5px solid #0F766E;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 10px 0;
        color: #334155;
        line-height: 1.65;
    }
    .recommendation.warning {
        border-left-color: #D97706;
    }
    .recommendation h4 {
        color: #172B4D;
        margin: 0 0 6px 0;
        font-size: 1rem;
    }
    .small-muted {
        color: #64748B;
        font-size: 0.83rem;
        line-height: 1.55;
    }
    div.stButton > button, div.stDownloadButton > button {
        background: #0F766E;
        color: white;
        border: 1px solid #0F766E;
        border-radius: 9px;
        font-weight: 650;
        min-height: 42px;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: #115E59;
        color: white;
        border-color: #115E59;
    }
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #94A3B8;
        border-radius: 12px;
        padding: 12px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
    }
    h1, h2, h3, h4, p, label {
        color: #172B4D;
    }
    hr {
        border-color: #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Resolve model location
# Expected repository structure:
# AI_Food_Waste_Prediction/
#   models/food_waste_model.pkl
#   streamlit_app/app.py
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "food_waste_model.pkl"


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🍎 Food Waste AI")
    st.caption("INVENTORY INTELLIGENCE")
    st.divider()

    st.markdown("### How to use")
    st.markdown(
        """
        1. Upload the preprocessed feature CSV.
        2. Check the input columns.
        3. Generate predictions.
        4. Review the metrics and charts.
        5. Read the suggested next actions.
        6. Download the prediction results.
        """
    )

    st.divider()
    st.markdown("### Built with")
    st.markdown(
        "<p class='small-muted'>Python · Pandas · Scikit-learn · "
        "Plotly · Streamlit</p>",
        unsafe_allow_html=True,
    )
    st.caption("Predictions are estimates, not guarantees.")

# ============================================================
# Hero header
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🍎 Food Waste Prediction &amp; Inventory Intelligence</h1>
        <p>
            Turn preprocessed inventory data into clear waste predictions.
            Explore overall estimates, identify records that may need review,
            and support better inventory decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Load model with clear errors
# ============================================================
if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at `{MODEL_PATH}`. "
        "Check that `models/food_waste_model.pkl` is committed to GitHub."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "The model could not be loaded. Check that the model file is valid "
        "and that the required package versions are compatible."
    )
    st.exception(exc)
    st.stop()

# ============================================================
# Upload input data
# ============================================================
st.markdown("<div class='section-title'>1. Upload your dataset</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-help'>Upload the same numeric, preprocessed feature "
    "CSV used by the model during training. Do not upload a raw CSV unless "
    "it has been processed into the exact training features.</div>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose X_preprocessed.csv",
    type=["csv"],
    help="Use the same feature columns and preprocessing as model training.",
)

if uploaded_file is None:
    st.info(
        "Upload your preprocessed CSV to begin. After prediction, this page "
        "will show summary metrics, charts, records for review, recommendations, "
        "and a downloadable results file."
    )
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if data.empty:
    st.warning("The uploaded CSV has no rows. Please upload a dataset containing records.")
    st.stop()

if data.select_dtypes(include=["object", "category"]).shape[1] > 0:
    st.warning(
        "Some columns contain text values. The model may require numeric, "
        "preprocessed features. Upload the encoded feature CSV used during training."
    )

st.success(f"Dataset loaded: {len(data):,} rows and {len(data.columns):,} columns.")

with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(data.head(10), use_container_width=True)
    st.caption("Showing the first 10 rows only.")

# ============================================================
# Predict
# ============================================================
st.markdown("<div class='section-title'>2. Generate predictions</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-help'>The model estimates waste quantity for each "
    "uploaded record. Review the outputs before making inventory changes.</div>",
    unsafe_allow_html=True,
)

if not st.button("Generate food waste predictions", type="primary", use_container_width=False):
    st.stop()

try:
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and len(data.columns) != expected_features:
        st.error(
            f"The uploaded data has {len(data.columns)} columns, but the model "
            f"expects {expected_features}. Upload the exact preprocessed feature "
            "set used during training."
        )
        st.stop()

    predictions = model.predict(data)
    result = data.copy()
    result["Predicted_Waste_Quantity"] = predictions

    waste_col = "Predicted_Waste_Quantity"
    result[waste_col] = pd.to_numeric(result[waste_col], errors="coerce")
    result = result.dropna(subset=[waste_col])

    if result.empty:
        st.error("The model did not return any valid predictions for these records.")
        st.stop()

except Exception as exc:
    st.error(
        "Prediction failed. Verify that the CSV has the exact features, "
        "encoding, and column order expected by the trained model."
    )
    st.exception(exc)
    st.stop()

# ============================================================
# Prediction summary cards
# ============================================================
pred = result[waste_col]
total_waste = float(pred.sum())
average_waste = float(pred.mean())
max_waste = float(pred.max())
records_count = len(result)
high_threshold = float(pred.quantile(0.90))
high_count = int((pred >= high_threshold).sum())

st.markdown("<div class='section-title'>3. Prediction summary</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-help'>These values summarize the model's estimates "
    "for the uploaded records. They are predicted quantities, not confirmed "
    "real-world waste.</div>",
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metrics = [
    ("Total predicted waste", f"{total_waste:,.2f}", "Sum across uploaded records"),
    ("Average per record", f"{average_waste:,.2f}", "Mean predicted quantity"),
    ("Highest prediction", f"{max_waste:,.2f}", "Largest record-level estimate"),
    ("Records analysed", f"{records_count:,}", "Rows with valid predictions"),
]
for col, (label, value, note) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">{label}</div>
                <div class="info-value">{value}</div>
                <div class="info-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# ============================================================
# Charts
# ============================================================
st.markdown("<div class='section-title'>4. Explore the predictions</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-help'>Use these charts to understand how the predicted "
    "values are distributed. The charts describe model output; they do not prove "
    "why waste occurs.</div>",
    unsafe_allow_html=True,
)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    hist = px.histogram(
        result,
        x=waste_col,
        nbins=25,
        title="Distribution of predicted waste",
        labels={waste_col: "Predicted waste quantity", "count": "Number of records"},
        color_discrete_sequence=["#0F766E"],
    )
    hist.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font_color="#172B4D",
        title_font_color="#172B4D",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(hist, use_container_width=True)

with chart_col2:
    bins = pd.cut(pred, bins=5, duplicates="drop")
    band_counts = bins.value_counts(sort=False).reset_index()
    band_counts.columns = ["Prediction range", "Records"]
    band_counts["Prediction range"] = band_counts["Prediction range"].astype(str)
    band_chart = px.bar(
        band_counts,
        x="Prediction range",
        y="Records",
        title="Records by prediction range",
        labels={"Prediction range": "Predicted waste range", "Records": "Number of records"},
        color_discrete_sequence=["#3B82A0"],
    )
    band_chart.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font_color="#172B4D",
        title_font_color="#172B4D",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_tickangle=-25,
    )
    st.plotly_chart(band_chart, use_container_width=True)

# If there are useful categorical columns, show group-level estimates.
categorical_candidates = [
    col for col in ["City", "Category", "Store_Type", "Region", "Store_ID", "Store_Name"]
    if col in result.columns
]
if categorical_candidates:
    group_by = st.selectbox(
        "Compare predicted waste by",
        options=categorical_candidates,
        help="Choose a column available in the uploaded dataset.",
    )
    grouped = (
        result.groupby(group_by, dropna=False)[waste_col]
        .agg(["sum", "mean", "count"])
        .reset_index()
        .sort_values("sum", ascending=False)
        .head(15)
    )
    grouped[group_by] = grouped[group_by].astype(str)
    grouped_chart = px.bar(
        grouped,
        x=group_by,
        y="sum",
        hover_data={"mean": ":.2f", "count": True, "sum": ":.2f"},
        title=f"Highest total predicted waste by {group_by}",
        labels={"sum": "Total predicted waste", group_by: group_by, "mean": "Average prediction", "count": "Records"},
        color_discrete_sequence=["#0F766E"],
    )
    grouped_chart.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font_color="#172B4D",
        title_font_color="#172B4D",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_tickangle=-25,
    )
    st.plotly_chart(grouped_chart, use_container_width=True)

# ============================================================
# High-prediction records
# ============================================================
st.markdown("<div class='section-title'>5. Records to review</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-help'>The following are the top 10% of records by "
    "predicted waste in this uploaded file. This is a relative review threshold, "
    "not a food-safety classification.</div>",
    unsafe_allow_html=True,
)

high_records = result[result[waste_col] >= high_threshold].sort_values(waste_col, ascending=False)
display_cols = []
for col in ["Store_Name", "Store_ID", "City", "Category", "Inventory_Remaining", "Shelf_Life_Days"]:
    if col in high_records.columns:
        display_cols.append(col)
display_cols.append(waste_col)

st.caption(
    f"Review threshold for this upload: {high_threshold:,.2f} predicted units "
    f"({high_count:,} records meet or exceed it)."
)
st.dataframe(
    high_records[display_cols].head(20),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# Plain-language interpretation and recommendations
# ============================================================
st.markdown("<div class='section-title'>6. What does this mean?</div>", unsafe_allow_html=True)

if average_waste == 0:
    interpretation = (
        "The model predicted zero average waste for these records. "
        "Confirm that the input data and preprocessing are correct before relying on this result."
    )
elif high_count > 0:
    interpretation = (
        f"The model estimates {total_waste:,.2f} total waste units across "
        f"{records_count:,} records, with an average of {average_waste:,.2f} units "
        f"per record. Start by reviewing the {min(10, len(high_records))} highest-prediction "
        "records shown above."
    )
else:
    interpretation = (
        f"The model estimates {total_waste:,.2f} total waste units across "
        f"{records_count:,} records. Review the summary and distribution charts "
        "to understand the range of estimates."
    )

st.markdown(
    f"""
    <div class="recommendation">
        <h4>📌 Plain-language summary</h4>
        {interpretation}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Suggested next actions")
st.markdown(
    "<p class='small-muted'>These are general inventory-management suggestions, "
    "not causes identified by the model.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="recommendation warning">
        <h4>1. Review high-prediction records</h4>
        Check stock levels, sales trends, expiry dates, and recent demand for
        the records listed above before taking action.
    </div>
    <div class="recommendation">
        <h4>2. Check upcoming expiry dates</h4>
        Where expiry information is available, prioritise safe, eligible stock
        rotation and follow applicable food-safety procedures.
    </div>
    <div class="recommendation">
        <h4>3. Compare predictions with actual outcomes</h4>
        Track observed waste over time and compare it with the model estimates
        to evaluate whether predictions are useful.
    </div>
    <div class="recommendation">
        <h4>4. Adjust inventory carefully</h4>
        Use verified sales and demand data, operational constraints, and staff
        judgement before changing order quantities.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Full results and download
# ============================================================
st.markdown("<div class='section-title'>7. Prediction results</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-help'>Review the full output table or download a CSV "
    "containing the uploaded features and the predicted waste quantity.</div>",
    unsafe_allow_html=True,
)

st.dataframe(result, use_container_width=True, hide_index=True)

csv_bytes = result.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download prediction results (CSV)",
    data=csv_bytes,
    file_name="food_waste_predictions.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown(
    "<p class='small-muted'>Important: Predictions are generated by a trained "
    "machine-learning model and should be treated as estimates. Confirm input "
    "features and validate predictions against real outcomes before operational use.</p>",
    unsafe_allow_html=True,
)
