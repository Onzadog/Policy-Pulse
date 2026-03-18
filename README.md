# PolicyPulse

PolicyPulse is a portfolio-friendly Streamlit project that lets users upload a CSV, generate a quick data profile, visualize numeric distributions, and fit a simple baseline model for regression or classification.


## Features

- Upload any CSV file
- Automatic dataset profiling
- Missing-value summary
- Target variable selection
- Automatic regression vs classification detection
- Baseline modeling with scikit-learn
- Quick narrative insight generation
- Downloadable column summary

## Why this is a good GitHub project

This repo shows that you can:

- build a complete Python app
- work with pandas and scikit-learn
- create visualizations with matplotlib
- package a project with a clear README and requirements file
- ship a project that someone else can run in a few minutes

## Tech stack

- Python
- Streamlit
- pandas
- scikit-learn
- matplotlib
- numpy

## Run locally

```bash
git clone <your-repo-url>
cd policypulse
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Suggested GitHub description

A Streamlit app for quick policy and business dataset profiling, visualization, and baseline forecasting.

## ways to improve it

- Add model comparison across random forest, XGBoost, and regularized linear models
- Export a PDF report
- Add correlation heatmaps and feature importance
- Add time-series forecasting support
- Deploy it on Streamlit Community Cloud

