import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path
import numpy as np
import hashlib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Food Waste Intelligence",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# POWER-BI-INSPIRED DASHBOARD THEME
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #F4F6F9; color: #172B4D; }
    [data-testid="stHeader"] { background: #102A4C; }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stSidebar"] { background: #102A4C; border-right: 3px solid #1B456D; }
    [data-testid="stSidebar"] * { color: #F4F7FB !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] { background: #FFFFFF; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] * { color: #172B4D !important; }
    .block-container { max-width: 1600px; padding-top: 1.2rem; padding-bottom: 2.5rem; }
    .hero { background: #102A4C; border: 1px solid #183C63; border-radius: 4px; padding: 21px 26px; margin-bottom: 18px; box-shadow: 0 4px 12px rgba(16,42,76,.12); }
    .hero-kicker { color: #BFD9F2 !important; font-size: .72rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 7px; }
    .hero h1 { color: #FFFFFF !important; font-size: clamp(1.55rem, 2.6vw, 2.15rem); line-height: 1.2; font-weight: 800; margin: 0 0 8px 0; }
    .hero p { color: #E0EAF5 !important; font-size: .92rem; line-height: 1.55; margin: 0; max-width: 1100px; }
    .section-title { color: #173A60 !important; font-size: 1.12rem; font-weight: 800; margin: 17px 0 5px 0; border-left: 4px solid #1E9B8A; padding-left: 10px; }
    .section-help { color: #5A6A7E !important; font-size: .84rem; line-height: 1.55; margin-bottom: 12px; }
    .metric-card { background: #FFFFFF; border: 1px solid #D8E0E9; border-top: 4px solid #1E9B8A; border-radius: 3px; padding: 14px 16px; min-height: 112px; box-shadow: 0 2px 7px rgba(16,42,76,.05); }
    .metric-label { color: #52647A !important; font-size: .76rem; font-weight: 700; margin-bottom: 8px; }
    .metric-value { color: #173A60 !important; font-size: clamp(1.3rem, 2vw, 1.75rem); font-weight: 800; line-height: 1.15; overflow-wrap: anywhere; }
    .metric-note { color: #64748B !important; font-size: .72rem; margin-top: 7px; line-height: 1.4; }
    .recommendation { background: #FFFFFF; border: 1px solid #D8E0E9; border-left: 4px solid #1E9B8A; border-radius: 3px; padding: 13px 16px; margin: 9px 0; color: #334155 !important; line-height: 1.6; }
    .recommendation * { color: #334155 !important; }
    .recommendation h4 { color: #173A60 !important; font-size: .94rem; font-weight: 750; margin: 0 0 5px 0; }
    .recommendation.warning { border-left-color: #E0A12B; background: #FFFCF5; }
    .recommendation.blue { border-left-color: #3C78B5; }
    .pill { display: inline-block; padding: 4px 9px; border-radius: 3px; background: #E1F3EF; color: #126D63 !important; font-size: .73rem; font-weight: 750; margin-bottom: 8px; }
    .muted { color: #64748B !important; font-size: .79rem; line-height: 1.55; }
    div.stButton > button, div.stDownloadButton > button { background: #167F78; color: #FFFFFF; border: 1px solid #167F78; border-radius: 4px; font-weight: 700; min-height: 40px; }
    div.stButton > button:hover, div.stDownloadButton > button:hover { background: #12665F; color: #FFFFFF; border-color: #12665F; }
    [data-testid="stFileUploader"] { background: #FFFFFF; border: 1px dashed #9AAABD; border-radius: 4px; padding: 10px; }
    [data-testid="stDataFrame"] { border: 1px solid #D8E0E9; border-radius: 3px; overflow: hidden; }
    div[data-testid="stAlert"] { border-radius: 4px; }
    h1, h2, h3, h4, p, label { color: #172B4D; }
    hr { border-color: #D8E0E9; }
    .stTabs [data-baseweb="tab-list"] { gap: 3px; background: #E8EDF3; padding: 4px; border-radius: 3px; }
    .stTabs [data-baseweb="tab"] { background: #F8FAFC; border-radius: 2px; color: #173A60; font-weight: 700; padding: 8px 14px; }
    .stTabs [aria-selected="true"] { background: #167F78 !important; color: #FFFFFF !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PATHS AND MODEL
# Expected repository:
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
# SIDEBAR — navigation and dataset upload
# ============================================================
with st.sidebar:
    st.markdown("## 🍎 Food Waste AI")
    st.markdown(
        "<span style='color:#7FE0D0;font-weight:800;font-size:.70rem;letter-spacing:.12em;'>INVENTORY INTELLIGENCE</span>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### Dashboard navigation")
    st.markdown("""
    **01**  Overview  

    **02**  Waste analytics  

    **03**  Store / product comparison  

    **04**  Records to review  

    **05**  Export results
    """)
    st.divider()
    st.markdown("### Upload model-ready data")
    uploaded_file = st.file_uploader(
        "Choose X_preprocessed.csv",
        type=["csv"],
        help="Upload numeric, preprocessed features matching the model's training inputs.",
        key="food_waste_csv_upload",
    )
    st.divider()
    st.markdown("### Project stack")
    st.markdown("<p class='muted'>Python · Pandas · Scikit-learn · Plotly · Streamlit</p>", unsafe_allow_html=True)
    st.markdown("<p class='muted'>Estimates are decision-support outputs, not verified measurements of actual waste.</p>", unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">FOOD WASTE • INVENTORY ANALYTICS</div>
        <h1>🍎 Food Waste Prediction &amp; Smart Inventory</h1>
        <p>
            Explore model-based waste estimates, compare records, and identify inventory entries for review.
            Upload the same preprocessed feature columns used to train the model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div style='background:#FFFFFF;border:1px solid #D8E0E9;border-left:5px solid #167F78;padding:9px 14px;margin:0 0 15px 0;color:#52647A;font-size:.82rem;'>DASHBOARD VIEW &nbsp; | &nbsp; Prediction summary &nbsp; • &nbsp; Analytics &nbsp; • &nbsp; Priority review &nbsp; • &nbsp; Export</div>", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Expected `models/food_waste_model.pkl` "
        "at the repository root. Confirm the file is committed to GitHub."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "The trained model could not be loaded. Check that the model file "
        "is valid and its required Python package versions are installed."
    )
    st.exception(exc)
    st.stop()

# ============================================================
# UPLOAD STATUS
# ============================================================
if uploaded_file is None:
    st.markdown("<div class='section-title'>Dashboard overview</div>", unsafe_allow_html=True)
    st.info("Upload your model-ready CSV from the left sidebar to populate the dashboard.")
    st.markdown("""
    <div class="recommendation blue"><h4>How to use this dashboard</h4>
    1. Upload a numeric, preprocessed CSV in the sidebar. <br>
    2. Generate predictions when the file is ready. <br>
    3. Explore KPIs, charts, high-prediction records, and downloadable results.</div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read the uploaded CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV has no rows. Please choose a dataset with records.")
    st.stop()

if data.columns.duplicated().any():
    st.error("The uploaded CSV contains duplicate column names. Please correct it and upload again.")
    st.stop()

# Model-ready input should be numeric and finite.
non_numeric = [
    col for col in data.columns
    if not pd.api.types.is_numeric_dtype(data[col])
]
if non_numeric:
    st.error(
        "The uploaded CSV contains non-numeric columns: "
        + ", ".join(map(str, non_numeric))
        + ". Upload the numeric, preprocessed feature dataset used for training."
    )
    st.stop()

if data.isna().any().any():
    st.error(
        "The uploaded CSV contains missing values. Apply the same missing-value "
        "preprocessing used during training before uploading."
    )
    st.stop()

try:
    numeric_values = data.to_numpy(dtype=float)
except (TypeError, ValueError):
    st.error("Some values cannot be converted to numeric values.")
    st.stop()

if not np.isfinite(numeric_values).all():
    st.error("The uploaded CSV contains infinite or invalid numeric values.")
    st.stop()

# ============================================================
# PREDICT — preserve results across Streamlit reruns
# ============================================================
st.markdown(
    "<div class='section-title'>Generate predictions</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='section-help'>Generate an estimate for every uploaded record. "
    "After predictions are generated, searching and interacting with the dashboard "
    "will not require you to generate them again.</div>",
    unsafe_allow_html=True,
)

# Identify the current upload so old predictions are never shown for a new file.
upload_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
if st.session_state.get("food_waste_upload_signature") != upload_signature:
    st.session_state.pop("food_waste_result", None)
    st.session_state["food_waste_upload_signature"] = upload_signature

if st.button("Generate food waste predictions", type="primary", key="generate_predictions"):
    try:
        expected_count = getattr(model, "n_features_in_", None)
        if expected_count is not None and len(data.columns) != int(expected_count):
            st.error(
                f"Feature-count mismatch: the uploaded file has {len(data.columns)} "
                f"columns, but the model expects {expected_count}. Upload the exact "
                "preprocessed feature set used during training."
            )
            st.stop()

        expected_names = getattr(model, "feature_names_in_", None)
        if expected_names is not None:
            expected_names = list(expected_names)
            missing_features = [name for name in expected_names if name not in data.columns]
            unexpected_features = [name for name in data.columns if name not in expected_names]
            if missing_features or unexpected_features:
                if missing_features:
                    st.error("Missing model features: " + ", ".join(map(str, missing_features)))
                if unexpected_features:
                    st.error("Unexpected input features: " + ", ".join(map(str, unexpected_features)))
                st.stop()
            prediction_input = data[expected_names]
        else:
            prediction_input = data

        with st.spinner("Generating predictions..."):
            predictions = model.predict(prediction_input)

        predictions = np.asarray(predictions).reshape(-1)
        if len(predictions) != len(data):
            st.error("The number of predictions did not match the number of uploaded records.")
            st.stop()
        if not np.isfinite(predictions.astype(float)).all():
            st.error("The model returned invalid prediction values.")
            st.stop()

        result = data.copy()
        result["Predicted_Waste_Quantity"] = predictions
        st.session_state["food_waste_result"] = result
        st.session_state["food_waste_result_signature"] = upload_signature
        st.success(f"Predictions generated for {len(result):,} records.")

    except Exception as exc:
        st.error(
            "Prediction failed. Confirm that the uploaded columns, feature order, "
            "and preprocessing match the model's training data."
        )
        st.exception(exc)

# Restore the result on every rerun, including when the user types in search.
result = st.session_state.get("food_waste_result")
result_signature = st.session_state.get("food_waste_result_signature")
if result is None or result.empty or result_signature != upload_signature:
    st.info("Your file is ready. Click **Generate food waste predictions** once to view the dashboard.")
    st.stop()

# ============================================================
# RESULTS
# ============================================================
result = st.session_state.get("food_waste_result")
if result is None or result.empty:
    st.stop()

waste_col = "Predicted_Waste_Quantity"
pred = pd.to_numeric(result[waste_col], errors="coerce")
valid_mask = pred.notna() & np.isfinite(pred)
result = result.loc[valid_mask].copy()
pred = pred.loc[valid_mask]

if result.empty:
    st.error("No valid predictions are available to display.")
    st.stop()

total_waste = float(pred.sum())
average_waste = float(pred.mean())
maximum_waste = float(pred.max())
minimum_waste = float(pred.min())
highest_index = pred.idxmax()
high_threshold = float(pred.quantile(0.90))
high_records = result.loc[pred >= high_threshold].sort_values(waste_col, ascending=False)
high_count = len(high_records)

st.divider()
st.markdown(
    "<div class='section-title'>Key performance indicators</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='section-help'>A high-level summary of the model estimates "
    "for this upload. Units are inherited from the target variable used "
    "during model training.</div>",
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metrics = [
    ("Total predicted waste", f"{total_waste:,.2f}", "Sum of all estimates"),
    ("Average per record", f"{average_waste:,.2f}", "Mean estimated quantity"),
    ("Highest prediction", f"{maximum_waste:,.2f}", "Largest individual estimate"),
    ("Records analysed", f"{len(result):,}", "Valid prediction rows"),
]
for col, (label, value, note) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    "<p class='muted'>These figures are model estimates, not verified measurements "
    "of actual food waste.</p>",
    unsafe_allow_html=True,
)

# ============================================================
# CHARTS
# ============================================================
st.markdown(
    "<div class='section-title'>Food waste analytics</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='section-help'>Explore the range of predictions and compare "
    "records. Charts describe the model output; they do not establish the "
    "cause of waste.</div>",
    unsafe_allow_html=True,
)

chart_left, chart_right = st.columns(2)

with chart_left:
    hist = px.histogram(
        result,
        x=waste_col,
        nbins=25,
        title="Predicted waste distribution",
        labels={waste_col: "Predicted waste quantity", "count": "Number of records"},
        color_discrete_sequence=["#168F83"],
    )
    hist.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#172B4D", family="Inter"),
        title_font=dict(color="#172B4D", size=17),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    hist.update_xaxes(gridcolor="#E8EEF4")
    hist.update_yaxes(gridcolor="#E8EEF4")
    st.plotly_chart(hist, use_container_width=True)

with chart_right:
    if pred.nunique() < 2:
        band_counts = pd.DataFrame({"Prediction range": [f"{pred.iloc[0]:,.2f}"], "Number of records": [len(pred)]})
    else:
        bins = pd.cut(pred, bins=min(5, pred.nunique()), duplicates="drop")
        band_counts = bins.value_counts(sort=False).reset_index()
        band_counts.columns = ["Prediction range", "Number of records"]
        band_counts["Prediction range"] = band_counts["Prediction range"].astype(str)
    band_chart = px.bar(
        band_counts,
        x="Prediction range",
        y="Number of records",
        title="Records by prediction range",
        labels={"Prediction range": "Predicted waste range"},
        color_discrete_sequence=["#3D79B5"],
    )
    band_chart.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#172B4D", family="Inter"),
        title_font=dict(color="#172B4D", size=17),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_tickangle=-25,
    )
    band_chart.update_xaxes(gridcolor="#E8EEF4")
    band_chart.update_yaxes(gridcolor="#E8EEF4")
    st.plotly_chart(band_chart, use_container_width=True)

# Optional breakdown using columns that actually exist in the uploaded file.
group_candidates = [
    col for col in ["City", "Category", "Store_Type", "Region", "Store_ID", "Store_Name"]
    if col in result.columns
]
if group_candidates:
    st.markdown("#### Compare predictions by group")
    group_col = st.selectbox(
        "Group records by",
        group_candidates,
        help="Only columns found in your uploaded CSV are available here.",
    )
    grouped = (
        result.groupby(group_col, dropna=False)[waste_col]
        .agg(Total="sum", Average="mean", Records="count")
        .reset_index()
        .sort_values("Total", ascending=False)
        .head(15)
    )
    grouped[group_col] = grouped[group_col].astype(str)
    group_chart = px.bar(
        grouped,
        x=group_col,
        y="Total",
        hover_data={"Average": ":.2f", "Records": True, "Total": ":.2f"},
        title=f"Highest total predicted waste by {group_col}",
        labels={"Total": "Total predicted waste"},
        color_discrete_sequence=["#168F83"],
    )
    group_chart.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#172B4D", family="Inter"),
        title_font=dict(color="#172B4D", size=17),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_tickangle=-25,
    )
    group_chart.update_xaxes(gridcolor="#E8EEF4")
    group_chart.update_yaxes(gridcolor="#E8EEF4")
    st.plotly_chart(group_chart, use_container_width=True)

# ============================================================
# RECORDS TO REVIEW
# ============================================================
st.markdown(
    "<div class='section-title'>Priority records to review</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='section-help'>This table highlights the top 10% of predictions "
    "within the uploaded file. It is a relative review threshold, not a validated "
    "food-safety risk score.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<span class='pill'>Review threshold: {high_threshold:,.2f}</span>",
    unsafe_allow_html=True,
)
st.caption(
    f"{high_count:,} records meet or exceed the 90th-percentile prediction "
    "threshold for this upload."
)

review_columns = [
    col for col in [
        "Store_Name", "Store_ID", "City", "Category",
        "Inventory_Remaining", "Shelf_Life_Days", "Demand", "Inventory_Sold",
    ]
    if col in high_records.columns
]
review_columns.append(waste_col)

if review_columns:
    st.dataframe(
        high_records[review_columns].head(20),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.dataframe(
        high_records[[waste_col]].head(20),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# PLAIN LANGUAGE INTERPRETATION
# ============================================================
st.markdown(
    "<div class='section-title'>Interpretation and next steps</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="recommendation blue">
        <h4>What the total means</h4>
        The model estimates a combined waste quantity of
        <b>{total_waste:,.2f}</b> across <b>{len(result):,}</b> records.
        This is the sum of the model's predictions.
    </div>
    <div class="recommendation">
        <h4>What the average means</h4>
        The average estimate is <b>{average_waste:,.2f}</b> per record.
        Individual records may have higher or lower predictions.
    </div>
    <div class="recommendation warning">
        <h4>Where to start reviewing</h4>
        The highest prediction is <b>{maximum_waste:,.2f}</b>, in input row
        <b>{int(result.index.get_loc(highest_index)) + 1}</b>
        of the displayed result set. Check relevant inventory, sales, demand,
        and expiry information before deciding on any action.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("#### Suggested next actions")
st.markdown(
    "<p class='muted'>These are general inventory-management suggestions, "
    "not causes identified by the model.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="recommendation warning">
        <h4>1. Review high-prediction records</h4>
        Check stock levels, sales trends, demand, and expiry dates for records
        shown in the review table. Verify the information before acting.
    </div>
    <div class="recommendation">
        <h4>2. Check approaching expiry dates</h4>
        If expiry information is available, review products approaching expiry
        and follow applicable food-safety procedures.
    </div>
    <div class="recommendation">
        <h4>3. Compare predictions with actual outcomes</h4>
        Track observed waste and compare it with model estimates over time
        to evaluate the model on real outcomes.
    </div>
    <div class="recommendation">
        <h4>4. Plan inventory carefully</h4>
        Use verified sales and demand information, operational constraints,
        and staff judgement before changing order quantities.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FULL RESULTS + FILTER + DOWNLOAD
# ============================================================
st.markdown(
    "<div class='section-title'>Detailed records and export</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='section-help'>Search the table to inspect particular records. "
    "The CSV download includes every valid prediction, regardless of the table filter.</div>",
    unsafe_allow_html=True,
)

search_text = st.text_input(
    "Search results",
    placeholder="Search any displayed column value...",
)
display_result = result.copy()
if search_text.strip():
    mask = display_result.astype(str).apply(
        lambda column: column.str.contains(
            search_text.strip(), case=False, regex=False, na=False
        )
    ).any(axis=1)
    display_result = display_result.loc[mask]

st.caption(f"Showing {len(display_result):,} of {len(result):,} records.")
st.dataframe(display_result, use_container_width=True, hide_index=True)

csv_bytes = result.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download prediction results (CSV)",
    data=csv_bytes,
    file_name="food_waste_predictions.csv",
    mime="text/csv",
)

with st.expander("About this model and its limitations"):
    st.markdown(
        """
        **What the app does:** A trained Random Forest regression model
        estimates waste quantity from uploaded feature values.

        **How to interpret results:**
        - A high estimate indicates a record worth reviewing, not a confirmed loss.
        - Predictions do not independently identify why waste may occur.
        - Charts summarize estimates and are not evidence of real-world outcomes.
        - Recommendation cards are general suggestions, not automated decisions.

        **Before operational use:** Validate feature preparation, assess the model
        on suitable unseen data, compare predictions with observed waste, and
        check for target leakage in the training pipeline.
        """
    )

st.divider()
st.markdown(
    "<p class='muted' style='text-align:center;'>"
    "<b>Food Waste Intelligence</b> · Built with Python, Scikit-learn, "
    "Pandas, Plotly and Streamlit<br>"
    "Decision support only — validate estimates against actual outcomes."
    "</p>",
    unsafe_allow_html=True,
)
