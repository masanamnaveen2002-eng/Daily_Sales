from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


DATA_FILE = Path(__file__).resolve().parent / (
    "UPI_Daily_Statistics_june_2021_to_June_2026.csv"
)
REQUIRED_COLUMNS = {"Date", "Volume_Million", "Value_Crore"}
MODEL_NAMES = ("Linear regression", "Random forest", "Decision tree")

st.set_page_config(
    page_title="Daily Sales | UPI Payments",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes | None) -> pd.DataFrame:
    if file_bytes is None:
        data = pd.read_csv(DATA_FILE)
    else:
        from io import BytesIO

        data = pd.read_csv(BytesIO(file_bytes))

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in ("Volume_Million", "Value_Crore"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def make_model(
    model_name: str,
    numeric_columns: list[str],
    category_columns: list[str],
):
    transformers = []
    if numeric_columns:
        transformers.append(("numbers", StandardScaler(), numeric_columns))
    if category_columns:
        transformers.append(
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                category_columns,
            )
        )
    preprocessing = ColumnTransformer(transformers)

    if model_name == "Linear regression":
        estimator = LinearRegression()
    elif model_name == "Random forest":
        estimator = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        estimator = DecisionTreeRegressor(max_depth=10, random_state=42)

    return make_pipeline(preprocessing, estimator)


st.title("Daily Sales")
st.caption("Explore daily UPI payment activity and estimate transaction value.")

with st.sidebar:
    st.header("Explore")
    uploaded_file = st.file_uploader("Use another UPI CSV", type="csv")

try:
    raw_data = load_data(uploaded_file.getvalue() if uploaded_file else None)
except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

invalid_rows = int(
    raw_data[["Date", "Volume_Million", "Value_Crore"]].isna().any(axis=1).sum()
)
data = raw_data.dropna(subset=["Date", "Volume_Million", "Value_Crore"]).copy()
if data.empty:
    st.error("The dataset has no rows with valid dates, volume, and transaction values.")
    st.stop()
if invalid_rows:
    st.warning(f"Skipped {invalid_rows:,} row(s) with missing or invalid required values.")

data = data.sort_values("Date").reset_index(drop=True)

with st.sidebar:
    date_range = st.date_input(
        "Date range",
        value=(data["Date"].min().date(), data["Date"].max().date()),
        min_value=data["Date"].min().date(),
        max_value=data["Date"].max().date(),
    )
    selected_model = st.selectbox("Regression model", MODEL_NAMES)
    test_fraction = st.slider(
        "Holdout period",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        format="%.0f%%",
    )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

filtered = data.loc[
    data["Date"].dt.date.between(start_date, end_date)
].copy()
if filtered.empty:
    st.warning("No records fall within the selected date range.")
    st.stop()

first_col, second_col, third_col, fourth_col = st.columns(4)
first_col.metric("Transactions", f"{filtered['Volume_Million'].sum():,.1f}M")
second_col.metric("Total value", f"₹{filtered['Value_Crore'].sum():,.0f} Cr")
third_col.metric("Daily average value", f"₹{filtered['Value_Crore'].mean():,.1f} Cr")
fourth_col.metric("Days in range", f"{len(filtered):,}")

st.subheader("Payment activity")
trend_col, monthly_col = st.columns([1.6, 1])

with trend_col:
    st.markdown("**Daily transaction value**")
    daily_values = filtered.set_index("Date")[["Value_Crore"]].rename(
        columns={"Value_Crore": "Value (₹ crore)"}
    )
    st.line_chart(daily_values, width="stretch")

with monthly_col:
    st.markdown("**Monthly trend**")
    monthly_metric = st.selectbox(
        "Monthly measure",
        ("Value (₹ crore)", "Transactions (million)"),
        label_visibility="collapsed",
    )
    monthly_values = (
        filtered.assign(Month=filtered["Date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month")[["Volume_Million", "Value_Crore"]]
        .sum()
        .rename(
            columns={
                "Volume_Million": "Transactions (million)",
                "Value_Crore": "Value (₹ crore)",
            }
        )
    )
    st.bar_chart(monthly_values[[monthly_metric]], width="stretch")

st.subheader("Model: estimate daily transaction value")
st.caption(
    "The latest portion of the selected dataset is held out for testing, "
    "so the evaluation follows time order rather than mixing future and past records."
)

model_data = data.copy()
model_data["Year"] = model_data["Date"].dt.year
model_data["Month"] = model_data["Date"].dt.month
model_data["Day"] = model_data["Date"].dt.day
model_data["Day_of_year"] = model_data["Date"].dt.dayofyear
model_data["Days_since_start"] = (
    model_data["Date"] - model_data["Date"].min()
).dt.days

numeric_features = [
    "Volume_Million",
    "Year",
    "Month",
    "Day",
    "Day_of_year",
    "Days_since_start",
]
for optional_column in ("Quarter", "Week"):
    if optional_column in model_data.columns:
        model_data[optional_column] = pd.to_numeric(
            model_data[optional_column], errors="coerce"
        )
        numeric_features.append(optional_column)

category_features = [
    column
    for column in ("Weekday", "Day Type", "Financial Year")
    if column in model_data.columns
]
features = numeric_features + category_features
model_data = model_data.dropna(subset=features + ["Value_Crore"])
split_at = int(len(model_data) * (1 - test_fraction))

if split_at < 2 or len(model_data) - split_at < 1:
    st.info("At least three valid rows are needed to train and evaluate a model.")
else:
    train_data = model_data.iloc[:split_at]
    test_data = model_data.iloc[split_at:]
    model = make_model(selected_model, numeric_features, category_features)
    model.fit(train_data[features], train_data["Value_Crore"])
    predictions = model.predict(test_data[features])
    mae = mean_absolute_error(test_data["Value_Crore"], predictions)
    rmse = mean_squared_error(test_data["Value_Crore"], predictions) ** 0.5
    r2 = (
        r2_score(test_data["Value_Crore"], predictions)
        if len(test_data) > 1
        else float("nan")
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Mean absolute error", f"₹{mae:,.1f} Cr")
    metric_col2.metric("Root mean squared error", f"₹{rmse:,.1f} Cr")
    metric_col3.metric("R² score", f"{r2:.3f}" if pd.notna(r2) else "Not available")
    st.caption(
        f"Trained on {len(train_data):,} days through "
        f"{train_data['Date'].max():%d %b %Y}; tested on "
        f"{len(test_data):,} later days."
    )

    st.markdown("**Try an estimate**")
    with st.form("prediction_form"):
        prediction_date = st.date_input(
            "Date to estimate",
            value=(data["Date"].max() + pd.Timedelta(days=1)).date(),
            key="prediction_date",
        )
        prediction_volume = st.number_input(
            "Expected transaction volume (million)",
            min_value=0.0,
            value=float(data["Volume_Million"].tail(30).mean()),
            step=1.0,
        )
        submitted = st.form_submit_button("Estimate value")

    if submitted:
        estimate = pd.DataFrame(
            [
                {
                    "Volume_Million": prediction_volume,
                    "Year": prediction_date.year,
                    "Month": prediction_date.month,
                    "Day": prediction_date.day,
                    "Day_of_year": prediction_date.timetuple().tm_yday,
                    "Days_since_start": (
                        pd.Timestamp(prediction_date) - model_data["Date"].min()
                    ).days,
                    **(
                        {"Quarter": (prediction_date.month - 1) // 3 + 1}
                        if "Quarter" in numeric_features
                        else {}
                    ),
                    **(
                        {
                            "Week": pd.Timestamp(prediction_date).isocalendar().week
                        }
                        if "Week" in numeric_features
                        else {}
                    ),
                    **(
                        {"Weekday": pd.Timestamp(prediction_date).day_name()}
                        if "Weekday" in category_features
                        else {}
                    ),
                    **(
                        {
                            "Day Type": (
                                "Weekend"
                                if prediction_date.weekday() >= 5
                                else "Weekday"
                            )
                        }
                        if "Day Type" in category_features
                        else {}
                    ),
                    **(
                        {
                            "Financial Year": (
                                f"{prediction_date.year}-{prediction_date.year + 1}"
                                if prediction_date.month >= 4
                                else f"{prediction_date.year - 1}-{prediction_date.year}"
                            )
                        }
                        if "Financial Year" in category_features
                        else {}
                    ),
                }
            ]
        )
        value_estimate = model.predict(estimate[features])[0]
        st.success(
            f"Estimated daily transaction value: ₹{value_estimate:,.2f} crore"
        )

with st.expander("Data quality and records"):
    quality_col1, quality_col2, quality_col3 = st.columns(3)
    quality_col1.metric("Duplicate rows", f"{int(raw_data.duplicated().sum()):,}")
    quality_col2.metric("Missing cells", f"{int(raw_data.isna().sum().sum()):,}")
    quality_col3.metric("Source columns", f"{raw_data.shape[1]:,}")
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.download_button(
        "Download filtered records",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="upi_filtered_records.csv",
        mime="text/csv",
    )
