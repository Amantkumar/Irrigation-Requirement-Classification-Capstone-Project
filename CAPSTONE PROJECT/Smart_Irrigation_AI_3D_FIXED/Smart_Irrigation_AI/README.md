# 🌱 AgriFlow AI — Smart Irrigation Requirement Predictor

An attractive Streamlit-based machine-learning prototype for **Learn Depth Academy Track 1 – Problem 17: Irrigation Requirement Classification**.

## Features

- Modern dashboard-style Streamlit UI
- Soil moisture, temperature, humidity, crop stage and rainfall inputs
- Decision Tree classification model
- Prediction probability
- Model performance dashboard
- Confusion matrix
- Responsive layout
- Reproducible training script
- Local model storage

## Project Structure

```text
Smart_Irrigation_AI/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── irrigation_data.csv
└── model/
    ├── irrigation_model.pkl
    └── metrics.json
```

## Installation

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python train_model.py
```

## Start UI

```bash
streamlit run app.py
```

## Important Dataset Note

The training script generates a synthetic educational dataset because the supplied capstone statement specifies expected inputs but does not provide a dataset in the problem statement itself.

For an academic submission, replace the generated dataset with a credited external dataset if your mentor/instructor requires real-world data.

## Disclaimer

This is an educational prototype. It should not be used as the sole basis for real irrigation decisions.
