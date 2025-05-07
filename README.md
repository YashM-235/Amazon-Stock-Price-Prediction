# Amazon Stock Price Prediction

This repository presents a full pipeline for predicting Amazon (AMZN) stock prices using state-of-the-art time series forecasting methods. The project includes data collection, preprocessing, feature engineering, exploratory analysis, and implementation of multiple forecasting models, ranging from classical ARIMA to deep learning architectures and hybrid ensembles.

## Features

- **Data:** Daily Amazon stock prices (2005–2019) with 36 features, including OHLC, volume, technical indicators (RSI, SMA, EMA, MACD, Bollinger Bands), macroeconomic variables (currency pairs, interest rates, bank stocks, indices), and engineered features (lagged returns, rolling statistics)[4].
- **Preprocessing:** Outlier handling (Winsorization), normalization, and chronological train-test split for robust evaluation.
- **Exploratory Analysis:** Trend, volatility, and correlation analysis; stationarity checks (ADF test); decomposition and autocorrelation plots.
- **Models Implemented:**
  - ARIMA/SARIMA (baseline)
  - Prophet
  - LSTM and GRU neural networks
  - Transformer and Informer architectures
  - Hybrid GRU+XGBoost and ensemble stacking
- **Evaluation:** Multi-horizon forecasting (short, medium, long-term); metrics include RMSE and MAE; visual comparisons of actual vs. predicted prices.
- **Best Results:** Hybrid GRU+XGBoost achieved the lowest RMSE (0.013–0.014), outperforming traditional and standalone deep models. Transformer-based models excelled in long-horizon forecasts[4].

## Repository Structure

- `Stock_Predict_Amazon_final.ipynb` - Main notebook with code and results.
- `Final-Report-on-Predictive-Analysis-of-Amazon-Stock-Price-Movements.docx` - Detailed project report.
- `data.csv` - Dataset
- `Codes/` Code Progress till now.
- `Figures/` - Visualizations and plots.
- `README.md` - Project overview and instructions.

## Getting Started

1. Clone this repository.
2. Install dependencies (see notebook for requirements).
3. Run `Stock_Predict_Amazon_final.ipynb` to reproduce the analysis and results.
4. Refer to the final report for methodology and discussion.

## Key Results

| Model                | RMSE (Single-step) | RMSE (Multi-step) |
|----------------------|--------------------|-------------------|
| ARIMA                | 0.298              | -                 |
| Prophet              | 0.214              | -                 |
| LSTM                 | 0.041              | 0.016 (optimized) |
| GRU                  | 0.041              | -                 |
| Hybrid GRU+XGBoost   | 0.013              | 0.014             |
| Transformer          | 0.058              | 0.151 (30-day)    |

## Highlights

- **Hybrid and attention-based models** deliver state-of-the-art accuracy for Amazon stock prediction.
- **Feature engineering** and macroeconomic indicators significantly enhance model performance.
- **Ensemble approaches** are recommended for robust, real-world forecasting.

## Author

Yash Mehta, 
Year 3, BTech CSE (Data Science)
Bennett University

## License

For academic and educational use.

*For detailed methodology and results, see the included report and notebook files.*
