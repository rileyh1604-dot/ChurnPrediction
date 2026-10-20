import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from churn_pipeline import MODEL_PATH, TARGET, clean_data, load_data, preprocess

st.set_page_config(page_title="Telco Churn Dashboard", page_icon="📉", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_clean_data():
    return clean_data(load_data())


bundle = load_bundle()
df = load_clean_data()
model = bundle["model"]
threshold = bundle["threshold"]


def predict_proba(raw_df):
    X = preprocess(raw_df, bundle["feature_columns"], bundle["scaler"])
    return model.predict_proba(X)[:, 1]


st.title("📉 Telco Customer Churn Dashboard")
st.caption("Gradient boosted trees trained with SMOTE, with the decision threshold tuned for recall ≥ 0.70.")

metrics = bundle["metrics"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{len(df):,}")
col2.metric("Overall churn rate", f"{df[TARGET].mean():.1%}")
col3.metric("Test precision", f"{metrics['precision']:.3f}")
col4.metric("Test recall", f"{metrics['recall']:.3f}")

predict_tab, batch_tab, performance_tab, insights_tab = st.tabs(
    ["🔮 Predict a customer", "📂 Batch scoring", "📊 Model performance", "💡 Insights"]
)

# ---------------- Single customer ----------------
with predict_tab:
    st.subheader("Enter customer details")
    left, middle, right = st.columns(3)

    with left:
        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
        monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)
        total = st.number_input("Total charges ($)", 0.0, 10000.0, float(monthly * tenure), step=10.0)

    with middle:
        st.markdown("**Demographics**")
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
        dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)
        st.markdown("**Phone**")
        phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"]) if phone == "Yes" else "No phone service"

    with right:
        st.markdown("**Internet**")
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        addons = {}
        for addon in ["Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]:
            addons[addon] = st.selectbox(addon, ["No", "Yes"]) if internet != "No" else "No internet service"

    customer = pd.DataFrame([{
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet,
        **addons,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": payment,
        "Monthly Charges": monthly,
        "Total Charges": total,
    }])

    probability = predict_proba(customer)[0]
    st.divider()
    result_col, gauge_col = st.columns([1, 2])
    with result_col:
        st.metric("Churn probability", f"{probability:.1%}")
        if probability >= threshold:
            st.error(f"⚠️ High churn risk (threshold {threshold:.2f})")
        else:
            st.success(f"✅ Low churn risk (threshold {threshold:.2f})")
    with gauge_col:
        st.progress(float(probability))

# ---------------- Batch scoring ----------------
with batch_tab:
    st.subheader("Score a list of customers")
    st.write("Upload a CSV or Excel file that has the same columns as the original Telco dataset.")
    uploaded = st.file_uploader("Customer file", type=["csv", "xlsx"])
    use_sample = st.checkbox("Or use the training dataset as a demo")

    batch = None
    if uploaded is not None:
        batch = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        batch = clean_data(batch)
    elif use_sample:
        batch = df.copy()

    if batch is not None:
        try:
            batch["Churn Probability"] = predict_proba(batch)
        except KeyError as error:
            st.error(f"The file is missing a required column: {error}")
            st.stop()
        batch["High Risk"] = batch["Churn Probability"] >= threshold
        batch = batch.sort_values("Churn Probability", ascending=False)

        a, b = st.columns(2)
        a.metric("Customers scored", f"{len(batch):,}")
        b.metric("Flagged high risk", f"{batch['High Risk'].sum():,} ({batch['High Risk'].mean():.1%})")

        show_columns = [c for c in ["CustomerID", "Contract", "Tenure Months", "Monthly Charges",
                                    "Churn Probability", "High Risk"] if c in batch.columns]
        st.dataframe(
            batch[show_columns],
            use_container_width=True,
            column_config={"Churn Probability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
        )
        st.download_button("⬇️ Download scored file", batch.to_csv(index=False), "scored_customers.csv", "text/csv")

# ---------------- Model performance ----------------
with performance_tab:
    st.subheader("How the threshold trades precision for recall")
    y_test = bundle["y_test"]
    test_probabilities = bundle["test_probabilities"]

    chosen = st.slider("Decision threshold", 0.05, 0.95, float(round(threshold, 2)), 0.01)
    predictions = (test_probabilities >= chosen).astype(int)

    a, b, c = st.columns(3)
    a.metric("Precision", f"{precision_score(y_test, predictions, zero_division=0):.3f}")
    b.metric("Recall", f"{recall_score(y_test, predictions):.3f}")
    c.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    curve_col, matrix_col = st.columns(2)
    with curve_col:
        thresholds = np.arange(0.05, 0.96, 0.01)
        curve = pd.DataFrame({
            "Threshold": np.tile(thresholds, 2),
            "Score": [precision_score(y_test, test_probabilities >= t, zero_division=0) for t in thresholds]
                     + [recall_score(y_test, test_probabilities >= t) for t in thresholds],
            "Metric": ["Precision"] * len(thresholds) + ["Recall"] * len(thresholds),
        })
        lines = alt.Chart(curve, title="Precision and recall by threshold").mark_line().encode(
            x="Threshold:Q", y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])), color="Metric:N"
        )
        marker = alt.Chart(pd.DataFrame({"Threshold": [chosen]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="Threshold:Q")
        st.altair_chart(lines + marker, use_container_width=True)

    with matrix_col:
        matrix = confusion_matrix(y_test, predictions)
        cells = pd.DataFrame([
            {"Actual": actual, "Predicted": predicted, "Count": int(matrix[i, j])}
            for i, actual in enumerate(["Actually stayed", "Actually churned"])
            for j, predicted in enumerate(["Predicted stay", "Predicted churn"])
        ])
        base = alt.Chart(cells, title="Confusion matrix (test set)").encode(
            x=alt.X("Predicted:N", sort=None, title=None), y=alt.Y("Actual:N", sort=None, title=None)
        )
        heatmap = base.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None))
        labels = base.mark_text(fontSize=18).encode(
            text="Count:Q",
            color=alt.condition(alt.datum.Count > matrix.max() / 2, alt.value("white"), alt.value("black")),
        )
        st.altair_chart((heatmap + labels).properties(height=300), use_container_width=True)

# ---------------- Insights ----------------
with insights_tab:
    st.subheader("What drives churn?")
    importance = (
        pd.DataFrame({"Feature": bundle["feature_columns"], "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(15)
    )
    st.altair_chart(
        alt.Chart(importance, title="Top 15 feature importances").mark_bar().encode(
            x="Importance:Q", y=alt.Y("Feature:N", sort="-x", title=None)
        ),
        use_container_width=True,
    )

    segment = st.selectbox(
        "Churn rate by",
        ["Contract", "Payment Method", "Internet Service", "Tech Support", "Paperless Billing", "Senior Citizen"],
    )
    rates = (df.groupby(segment)[TARGET].mean() * 100).sort_values(ascending=False).reset_index()
    st.altair_chart(
        alt.Chart(rates, title=f"Churn rate by {segment}").mark_bar().encode(
            x=alt.X(f"{segment}:N", sort="-y"), y=alt.Y(f"{TARGET}:Q", title="Churn rate (%)")
        ),
        use_container_width=True,
    )

    tenure_bins = pd.cut(df["Tenure Months"], bins=[-1, 6, 12, 24, 48, 72],
                         labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    tenure_rates = (df.groupby(tenure_bins, observed=True)[TARGET].mean() * 100).reset_index()
    st.altair_chart(
        alt.Chart(tenure_rates, title="Churn rate by tenure (months)").mark_bar().encode(
            x=alt.X("Tenure Months:N", sort=None), y=alt.Y(f"{TARGET}:Q", title="Churn rate (%)")
        ),
        use_container_width=True,
    )
