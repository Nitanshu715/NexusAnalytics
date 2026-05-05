# ML Assignment — CO5
### UPES Dehradun | B.Tech CSE | Batch 2 CCVT

| Field | Details |
|---|---|
| Name | Nitanshu Tak |
| SAP ID | 500121943 |
| Batch | 2 CCVT |
| Course Outcome | CO5 — Unsupervised Learning & Time Series Analysis |
| University | University of Petroleum and Energy Studies, Dehradun |

---

## Overview

This repository contains the complete implementation for two machine learning lab questions under CO5:

**Question 1** applies Principal Component Analysis (PCA) to a customer purchase behaviour dataset to reduce dimensionality, determine the optimal number of components, and visualise customer segments in 2D and 3D space.

**Question 2** performs a full time series analysis pipeline on three years of daily retail sales data — including decomposition, stationarity testing, ARIMA modelling, and 30-day sales forecasting.

Both questions generate production-quality visualisations and a complete written report is included as a Word document.

---

## Repository Structure

```
ml-assignment/
├── q1_pca/
│   ├── pca_analysis.py          # Complete Q1 implementation
│   ├── plot_scree.png           # Scree plot + cumulative variance
│   ├── plot_pca_2d.png          # 2D PCA scatter (PC1 vs PC2)
│   ├── plot_pca_3d.png          # 3D PCA scatter (PC1, PC2, PC3)
│   └── plot_pca_loadings.png    # Feature loading bars for PC1 and PC2
│
├── q2_timeseries/
│   ├── timeseries_analysis.py   # Complete Q2 implementation
│   ├── plot_ts_raw.png          # Raw time series + monthly mean
│   ├── plot_decomposition.png   # Trend, seasonal, residual decomposition
│   ├── plot_acf_pacf.png        # ACF and PACF for model identification
│   ├── plot_forecast.png        # Actual vs ARIMA predicted (60-day test)
│   └── plot_30day_forecast.png  # 30-day forward forecast (Jan 2024)
│
├── report/
│   └── ML_Assignment_Report_Nitanshu_Tak.docx   # Full submission report
│
├── requirements.txt
└── README.md
```

---

## Question 1: Customer Behaviour Analysis using PCA

### What the Script Does

1. Generates a synthetic customer dataset with 500 records and 10 features (annual income, spending score, purchase frequency, average basket size, loyalty years, online purchase ratio, number of product categories, returns percentage, discount usage, and weekend shopping ratio).
2. Introduces 4% missing values to simulate real-world data quality issues.
3. Preprocesses the data using median imputation and StandardScaler normalisation.
4. Applies full PCA and extracts explained variance per component.
5. Plots a scree plot and cumulative variance curve to identify the optimal number of components.
6. Projects the data onto the top 2 and top 3 principal components and visualises results as coloured scatter plots.
7. Generates a feature loadings chart to interpret what each principal component represents.

### Key Findings

- 8 principal components are required to explain 80% of total variance, indicating the dataset has genuinely high intrinsic dimensionality.
- PC1 (16.37% variance) is dominated by Annual Income and Purchase Frequency — a composite "high-value customer" axis.
- PC2 (11.81% variance) captures Online Purchase Ratio vs in-store shopping and category breadth.
- The 2D projection shows partial income-based cluster separation along PC1.
- The 3D projection reveals additional separability for medium-income customers along PC3.

### Visualisations Produced

| Plot | Description |
|---|---|
| `plot_scree.png` | Explained variance per component (bar) and cumulative variance (line) with 80% and 95% thresholds |
| `plot_pca_2d.png` | Scatter of all 500 customers on PC1 vs PC2, coloured by income tertile |
| `plot_pca_3d.png` | Three-dimensional scatter on PC1, PC2, PC3 |
| `plot_pca_loadings.png` | Horizontal bar chart of feature contributions to PC1 and PC2 |

---

## Question 2: Retail Sales Forecasting using ARIMA

### What the Script Does

1. Generates 3 years of daily retail sales data (Jan 2021 – Dec 2023) with a realistic structure: linear upward trend, annual seasonality, day-of-week effects, festival spikes in October and December, and Gaussian noise.
2. Introduces 16 missing values (~1.5%) handled via time-based linear interpolation.
3. Plots the raw series with a monthly mean overlay.
4. Decomposes the series into trend, seasonal, and residual components using an additive model (period = 365 days).
5. Runs the Augmented Dickey-Fuller test. The original series is non-stationary (p = 0.689). Log transformation + first-order differencing achieves stationarity (p < 0.0001).
6. Plots ACF and PACF on the training set to identify model orders (p=2, d=1, q=2).
7. Fits ARIMA(2,1,2) to the log-transformed training data (first 1,035 observations).
8. Generates forecasts on a 60-day held-out test set and evaluates with MAE and RMSE.
9. Produces a 30-day forward forecast for January 2024 with 95% confidence intervals.

### Key Results

| Metric | Value |
|---|---|
| ADF p-value (original) | 0.689 — Non-stationary |
| ADF p-value (log-differenced) | < 0.0001 — Stationary |
| ARIMA Order | (2, 1, 2) |
| MAE (60-day test) | INR 2,066 |
| RMSE (60-day test) | INR 2,805 |
| MAPE | 14.26% |
| 30-day Forecast (Jan 2024) | INR 11,035 per day |

### Visualisations Produced

| Plot | Description |
|---|---|
| `plot_ts_raw.png` | Full 3-year daily sales series with monthly mean trend line |
| `plot_decomposition.png` | Four-panel additive decomposition (observed, trend, seasonal, residual) |
| `plot_acf_pacf.png` | ACF and PACF for model order identification |
| `plot_forecast.png` | Actual vs ARIMA(2,1,2) predicted sales on the 60-day test set |
| `plot_30day_forecast.png` | 30-day forward forecast for January 2024 with confidence interval |

---

## Setup and Installation

### Prerequisites

- Python 3.9 or above
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Question 1

```bash
cd q1_pca
python pca_analysis.py
```

All four plots are saved to the `q1_pca/` directory.

### Run Question 2

```bash
cd q2_timeseries
python timeseries_analysis.py
```

All five plots are saved to the `q2_timeseries/` directory. The 30-day forecast table is also printed to console.

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
statsmodels
scipy
```

Full pinned versions are in `requirements.txt`.

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, datetime handling, resampling |
| `numpy` | Numerical operations and synthetic data generation |
| `matplotlib` | All visualisations (2D, 3D, subplots) |
| `scikit-learn` | PCA, StandardScaler, SimpleImputer, MAE |
| `statsmodels` | seasonal_decompose, adfuller, ARIMA, ACF/PACF |
| `scipy` | Supporting statistical functions |

---

## Report

The full written report (`report/ML_Assignment_Report_Nitanshu_Tak.docx`) covers:

- Problem statement and dataset description for both questions
- Detailed methodology with justification for each step
- All visualisations with captions and interpretation
- Stationarity test results table
- Model evaluation metrics table
- Business insights and recommendations

---

## Notes

- Both scripts generate synthetic data internally — no external CSV file is required.
- Random seeds are fixed (`np.random.seed(42)`) for full reproducibility.
- All plots are saved as high-resolution PNG (150 DPI) in their respective directories.
- The ARIMA model is fitted on log-transformed data and back-transformed using `np.exp()` for interpretability.

---

*Submitted in partial fulfilment of CO5 — Machine Learning Laboratory, UPES Dehradun.*
