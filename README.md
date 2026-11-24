# Fraud Detection Streamlit App

## Run from the VS Code terminal (Windows)

Open the project folder in VS Code, choose **Terminal > New Terminal**, and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit, normally http://127.0.0.1:8501.
Leave the terminal open while using the app. Press **Ctrl+C** in that terminal to stop it.

The app includes a dataset dashboard, single-transaction scoring, and batch CSV
screening. Keep `app.py`, `fraud_detection_xgboost_model.pkl`, and
`Fraud_Analysis_Dataset.csv` in the same folder. The `.streamlit/config.toml`
configuration selects a light theme.

The saved model artifact is loaded with joblib. Only use a trusted model file.
Predictions are for educational review, not automated financial decisions.
