import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="PolicyPulse",
    page_icon="📊",
    layout="wide",
)

st.title("📊 PolicyPulse")
st.caption("Upload a CSV, inspect the data, and run a quick baseline model.")


# --------------------------------------------------
# Helpers
# --------------------------------------------------
@st.cache_data
def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load CSV data safely. Returns None if no file is provided."""
    if uploaded_file is None:
        return None

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Could not read the uploaded file as a CSV: {e}")
        return None


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact column summary table."""
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null_count": df.notna().sum().values,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean() * 100).round(2).values,
        "unique_values": df.nunique(dropna=True).values,
    })
    return summary


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude=np.number).columns.tolist()


def download_summary_csv(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    summarize_dataframe(df).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Upload")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.sidebar.header("Options")
show_raw_data = st.sidebar.checkbox("Show full preview table", value=False)


# --------------------------------------------------
# Load data
# --------------------------------------------------
df = load_data(uploaded_file)

if df is None:
    st.info("Upload a CSV file to begin.")
    st.stop()


# --------------------------------------------------
# Header metrics
# --------------------------------------------------
st.success("Dataset loaded successfully.")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", f"{len(df.columns):,}")
col3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")


# --------------------------------------------------
# Preview
# --------------------------------------------------
st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

if show_raw_data:
    st.subheader("Full Dataset")
    st.dataframe(df, use_container_width=True)


# --------------------------------------------------
# Column summary
# --------------------------------------------------
st.subheader("Column Summary")
summary_df = summarize_dataframe(df)
st.dataframe(summary_df, use_container_width=True)

st.download_button(
    label="Download Column Summary CSV",
    data=download_summary_csv(df),
    file_name="column_summary.csv",
    mime="text/csv",
)


# --------------------------------------------------
# Descriptive stats
# --------------------------------------------------
numeric_cols = get_numeric_columns(df)
categorical_cols = get_categorical_columns(df)

st.subheader("Descriptive Statistics")

if numeric_cols:
    st.write("**Numeric columns**")
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
else:
    st.warning("No numeric columns found.")

if categorical_cols:
    with st.expander("Categorical Summary"):
        cat_summary = pd.DataFrame({
            "column": categorical_cols,
            "non_null_count": [df[col].notna().sum() for col in categorical_cols],
            "unique_values": [df[col].nunique(dropna=True) for col in categorical_cols],
            "top_value": [
                df[col].mode(dropna=True).iloc[0]
                if not df[col].mode(dropna=True).empty
                else None
                for col in categorical_cols
            ],
        })
        st.dataframe(cat_summary, use_container_width=True)


# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.subheader("Visualization")

if not numeric_cols:
    st.warning("No numeric columns available for plotting.")
else:
    plot_type = st.selectbox(
        "Select plot type",
        ["Histogram", "Box Plot", "Scatter Plot"]
    )

    if plot_type in ["Histogram", "Box Plot"]:
        selected_col = st.selectbox("Select numeric column", numeric_cols)

        fig, ax = plt.subplots(figsize=(8, 5))

        if plot_type == "Histogram":
            ax.hist(df[selected_col].dropna(), bins=30)
            ax.set_title(f"Distribution of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frequency")

        elif plot_type == "Box Plot":
            ax.boxplot(df[selected_col].dropna(), vert=True)
            ax.set_title(f"Box Plot of {selected_col}")
            ax.set_ylabel(selected_col)

        st.pyplot(fig)

    elif plot_type == "Scatter Plot":
        if len(numeric_cols) < 2:
            st.warning("You need at least two numeric columns for a scatter plot.")
        else:
            x_col = st.selectbox("X-axis", numeric_cols, index=0)
            y_col = st.selectbox("Y-axis", numeric_cols, index=1)

            fig, ax = plt.subplots(figsize=(8, 5))
            plot_df = df[[x_col, y_col]].dropna()

            ax.scatter(plot_df[x_col], plot_df[y_col])
            ax.set_title(f"{y_col} vs. {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

            st.pyplot(fig)


# --------------------------------------------------
# Correlation
# --------------------------------------------------
st.subheader("Correlation Matrix")

if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr(numeric_only=True)
    st.dataframe(corr, use_container_width=True)
else:
    st.info("At least two numeric columns are needed for a correlation matrix.")


# --------------------------------------------------
# Baseline modeling
# --------------------------------------------------
st.subheader("Baseline Modeling")

if len(numeric_cols) < 2:
    st.warning("You need at least two numeric columns to run a baseline model.")
else:
    target = st.selectbox("Select target variable", numeric_cols)
    feature_options = [col for col in numeric_cols if col != target]
    features = st.multiselect(
        "Select feature variables",
        feature_options,
        default=feature_options[: min(3, len(feature_options))]
    )

    model_type = st.radio(
        "Select model type",
        ["Linear Regression", "Logistic Regression"],
        horizontal=True
    )

    if not features:
        st.info("Select at least one feature to run a model.")
    else:
        model_df = df[features + [target]].copy()

        X = model_df[features]
        y = model_df[target]

        if len(model_df) < 5:
            st.warning("Not enough rows to run a model.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
            )

            if model_type == "Linear Regression":
                pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ])

                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)

                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                m1, m2 = st.columns(2)
                m1.metric("Mean Squared Error", f"{mse:.4f}")
                m2.metric("R²", f"{r2:.4f}")

                results_df = pd.DataFrame({
                    "Actual": y_test.values,
                    "Predicted": preds,
                    "Residual": y_test.values - preds,
                })
                st.write("**Prediction Results**")
                st.dataframe(results_df.head(20), use_container_width=True)

                fitted_model = pipeline.named_steps["model"]
                coef_df = pd.DataFrame({
                    "feature": features,
                    "coefficient": fitted_model.coef_,
                }).sort_values(by="coefficient", key=lambda s: s.abs(), ascending=False)

                st.write("**Model Coefficients**")
                st.dataframe(coef_df, use_container_width=True)

            else:
                y_binary = (y > y.median()).astype(int)
                y_train_bin = y_binary.loc[X_train.index]
                y_test_bin = y_binary.loc[X_test.index]

                pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000)),
                ])

                pipeline.fit(X_train, y_train_bin)
                preds = pipeline.predict(X_test)

                acc = accuracy_score(y_test_bin, preds)
                cm = confusion_matrix(y_test_bin, preds)

                st.metric("Classification Accuracy", f"{acc:.4f}")

                st.write("**Confusion Matrix**")
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual 0", "Actual 1"],
                    columns=["Predicted 0", "Predicted 1"],
                )
                st.dataframe(cm_df, use_container_width=True)

                fitted_model = pipeline.named_steps["model"]
                coef_df = pd.DataFrame({
                    "feature": features,
                    "coefficient": fitted_model.coef_[0],
                }).sort_values(by="coefficient", key=lambda s: s.abs(), ascending=False)

                st.write("**Model Coefficients**")
                st.dataframe(coef_df, use_container_width=True)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption("PolicyPulse is a lightweight exploratory data app for quick profiling and baseline modeling.")
