# CIB Stock Price Forecasting (COMI)

A Machine Learning project focused on analyzing and forecasting the stock price of **Commercial International Bank (CIB)**, listed on the Egyptian Exchange (EGX), using historical market data.

## Project Overview

The goal of this project is to explore historical stock market data, identify patterns and relationships between market features, and build machine learning models to forecast the next trading day's closing price.

The project follows a data-driven approach, combining financial data analysis with predictive modeling.

## Objectives

- Analyze historical CIB stock market data.
- Explore relationships between Open, High, Low, and Close prices.
- Perform exploratory data analysis (EDA).
- Build and evaluate machine learning regression models.
- Forecast the next trading day's closing price.
- Compare model performance using regression evaluation metrics.

## Dataset

The dataset contains **1,265 daily observations** of CIB stock market data, covering the period from September 2021 to September 2026.

Main features:

| Feature | Description |
|---|---|
| Date | Trading date |
| Open | Opening price |
| High | Highest price |
| Low | Lowest price |
| Close | Closing price |
| Volume | Trading volume |
| Change % | Daily percentage change |

Additional engineered features include historical returns, technical indicators, volatility measures, and lagged closing prices.

## Exploratory Data Analysis (EDA)

The analysis revealed several interesting patterns in CIB's historical stock data:

1. **Stock Price Movement:** The closing price ranged from 0.964 (October 25, 2023) to 2.870 (February 18, 2026), showing substantial price movement throughout the analyzed period.

2. **Daily Return Distribution:** Positive-return days accounted for 46.36% of observations, negative-return days for 45.57%, and unchanged days for 8.07%, indicating a relatively balanced distribution between positive and negative daily movements.

3. **Volatility Patterns:** The average 20-day rolling volatility was approximately 2.22%, while the highest observed value reached 5.87% during March 2024, highlighting periods of elevated market volatility.

4. **Trading Volume Distribution:** Average daily trading volume was approximately 93,758 shares, compared to a median of 33,550 shares. The maximum recorded volume reached 3.14 million shares, indicating a highly right-skewed volume distribution.

5. **Price Autocorrelation:** The previous day's closing price showed a strong correlation of 0.994 with the next day's closing price. However, lagged closing prices had very weak correlations with next-day returns, ranging approximately from -0.02 to -0.03.

## Model Evaluation Results

Several machine learning regression models were experimented with and evaluated for next-day stock price forecasting.

The evaluation included regression metrics such as R² Score, MAE, and RMSE.

| Evaluation Metric | Result |
|---|---:|
| R² Score | **0.86** |

The model achieved an R² score of **0.86** on the test set, explaining approximately 86% of the variance in the actual closing prices.
