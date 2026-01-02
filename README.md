<h1 align="center"> Stock Price Prediction Using Machine Learning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python3-blue" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Stock%20Prediction-yellowgreen" />
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-orange" />
  <img src="https://img.shields.io/github/stars/dhakarshailendra829/StockPricePredictionUsingMachineLearning?style=social" />
</p>

---

## Project Overview
This project focuses on predicting stock prices using machine learning techniques applied to historical market data. It implements a complete end-to-end pipeline, covering data collection, preprocessing, feature engineering, model training, and evaluation.
The system is integrated with an interactive Streamlit dashboard that enables users to visualize historical trends and view model-generated price forecasts in real time. By combining statistical analysis, machine learning models, and an intuitive user interface, the project demonstrates the practical application of AI in financial market analysis and decision support.
---

## Purpose of the Project
The main objectives of this project are:
- To apply machine learning on stock market data  
- To predict future stock prices based on historical trends  
- To visualize predictions using an interactive dashboard  
- To demonstrate a complete ML pipeline from training to deployment  
This project is suitable for **college projects, portfolio building, and interview demonstrations**.
---

## Why This Project Is Important
Stock market prices are highly volatile and non-linear. Traditional statistical methods often fail to capture complex patterns in market data.  
Machine learning models help in:
- Identifying hidden trends in historical prices  
- Handling large datasets efficiently  
- Assisting in Buy / Sell / Hold decision analysis  
This project reflects how AI is used in **financial forecasting, trading systems, and decision-support tools**.
---

## How the System Works
1. Historical stock data is collected  
2. Features such as trends, averages, and volatility are generated  
3. Data is scaled using pre-trained scalers  
4. Machine learning models predict future prices  
5. Results are displayed through an interactive Streamlit dashboard  
---

## Repository Structure
| File / Folder | Description |
|---------------|-------------|
| `app.py` | Streamlit dashboard for prediction and visualization |
| `machinelearningadvancedstockproject.py` | Model training and feature engineering script |
| `production_xgb.pkl` | Trained XGBoost model |
| `production_lgb.pkl` | Trained LightGBM model |
| `production_lstm.keras` | Trained LSTM deep learning model |
| `production_ensemble.pkl` | Ensemble model combining multiple models |
| `scaler_features.pkl` | Feature scaler |
| `scaler_target.pkl` | Target scaler |
| `advanced_stock_data.csv` | Dataset used for training |

---

## Machine Learning Models Used
- XGBoost Regressor  
- LightGBM Regressor  
- LSTM Neural Network  
- Ensemble Model (combination of multiple models)  
These models are trained on historical stock data and stored for reuse in the application.
---

## Dashboard Features
- Stock ticker selection  
- Historical price visualization  
- Future price prediction  
- Buy / Sell / Hold trading signals  
- Prediction confidence percentage  
- Interactive Plotly charts  
- Professional dark-themed UI  
---

## Requirements
The project requires the following:
- Python 3.8 or higher  
- Streamlit  
- pandas  
- numpy  
- yfinance  
- scikit-learn  
- joblib  
- plotly  
- tensorflow / keras  
---

## How to Run the Project
1. Clone the repository  
   ```bash
   git clone https://github.com/dhakarshailendra829/StockPricePredictionUsingMachineLearning.git
   cd StockPricePredictionUsingMachineLearning
2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Run the application
    ```bash
    streamlit run app.py
---

## Use Cases
Academic machine learning projects
Learning financial data analysis
Stock market prediction demos
Portfolio and resume showcase
Understanding ML model deployment with UI 
---

## Acknowledgements
Yahoo Finance for providing historical stock data
Streamlit for the interactive dashboard framework
Open-source libraries such as scikit-learn, XGBoost, LightGBM, TensorFlow, and Plotly
---

## Contribution
If you find this project useful:
Star the repository
Report issues
Suggest improvements
Contributions and feedback are welcome.
---

## 👤 Author
Shailendra Dhakad
Machine Learning & AI Enthusiast
GitHub: https://github.com/dhakarshailendra829
