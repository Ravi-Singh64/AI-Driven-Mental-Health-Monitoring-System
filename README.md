# 🧠 AI-Driven Mental Health Monitoring System

This is a machine learning-powered Streamlit web application that predicts **Anxiety Score**, **Depression Score**, and **Stress Level** based on user input from behavioral, physical, and social media indicators.

## 🚀 Live Demo

👉 [Click here to use the app](https://ai-driven-mental-health-monitoring-system-aeemfm5u6kr74muypcxl.streamlit.app/) 


---

## 📋 Features

- Predicts:
  - Anxiety Score & Level
  - Depression Score & Level
  - High Stress Risk
- Text sentiment analysis from:
  - Posts
  - Comments
  - Status updates
- Derived features like mood-stress ratio, activity level, etc.
- Uses compressed `.joblib` models for faster loading
- Clean and responsive Streamlit UI

---

## 📦 Technologies Used

- **Python**
- **Streamlit** (for UI)
- **scikit-learn** (for ML models)
- **Joblib** (for compressed model storage)
- **VADER Sentiment Analysis** (for text scoring)

---

## 🗂️ Project Structure

mental-health-monitoring-system/
├── app.py
├── requirements.txt
├── linear_regression_anxiety_score.pkl
├── linear_regression_depression_score.pkl
├── logistic_regression_high_stress.pkl
├── encoder_anxiety_level.pkl
├── encoder_depression_level.pkl
├── minmax_scaler.pkl
├── scaled_columns.pkl
├── model_feature_columns.pkl
├── random_forest_anxiety_level.joblib
├── random_forest_depression_level.joblib
└── README.md



---

## 🧪 Run Locally

```bash
# Clone the repo
git clone https://github.com/Ravi-Singh64/AI-Driven-Mental-Health-Monitoring-System.git
cd AI-Driven-Mental-Health-Monitoring-System

# Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


## Other details
🙋‍♂️ Author
👤 Ravi Singh

GitHub: @Ravi-Singh64


📃 License
This project is licensed under the MIT License — feel free to use and modify!

🚧 This project is part of a larger AI-based wellness system. More features like notifications, mobile support, and dashboards will be added in the future.
