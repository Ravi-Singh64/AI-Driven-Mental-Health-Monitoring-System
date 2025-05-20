import streamlit as st
import numpy as np
import pandas as pd
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==========================
# ✅ Load Models and Tools
# ==========================

import joblib  

# Load joblib-compressed RandomForest models
clf_anxiety = joblib.load("random_forest_anxiety_level.joblib")
clf_depression = joblib.load("random_forest_depression_level.joblib")



with open("linear_regression_anxiety_score.pkl", "rb") as f:
    reg_anxiety = pickle.load(f)
with open("linear_regression_depression_score.pkl", "rb") as f:
    reg_depression = pickle.load(f)

with open("logistic_regression_high_stress.pkl", "rb") as f:
    clf_stress = pickle.load(f)
with open("encoder_anxiety_level.pkl", "rb") as f:
    enc_anx = pickle.load(f)
with open("encoder_depression_level.pkl", "rb") as f:
    enc_dep = pickle.load(f)
with open("minmax_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("scaled_columns.pkl", "rb") as f:
    scaled_columns = pickle.load(f)
with open("model_feature_columns.pkl", "rb") as f:
    model_feature_columns = pickle.load(f)

analyzer = SentimentIntensityAnalyzer()

# ==========================
# ✅ Streamlit UI
# ==========================
st.set_page_config(page_title="Mental Health Monitoring", layout="wide")
st.title("🧠 AI-Powered Mental Health Monitor")
st.markdown("Fill the inputs below to estimate mental wellness indicators.")

col1, col2 = st.columns(2)
with col1:
    heart_rate = st.slider("Heart Rate (bpm)", 40, 120, 72)
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, step=0.5)
    activity_level = st.slider("Activity Level (steps)", 0, 20000, 4000)
    body_temp = st.slider("Body Temperature (°F)", 95.0, 104.0, 98.6, step=0.1)
    mood = st.slider("Mood (1-10)", 1, 10, 6)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    calendar_events = st.slider("Calendar Events Today", 0, 10, 3)
    posts = st.text_area("Posts With Text", "")
with col2:
    daily_reflections = st.slider("Daily Reflections", 0, 10, 3)
    social_posts = st.slider("Social Posts", 0, 20, 4)
    interaction_freq = st.slider("Interaction Frequency", 0, 100, 30)
    screen_time = st.slider("Screen Time (hrs)", 0.0, 24.0, 6.0, step=0.5)
    location_changes = st.slider("Location Changes", 0, 10, 3)
    comments = st.text_area("Comments On Posts", "")
    status = st.text_area("Status Updates", "")

# ==========================
# ✅ Sentiment Analysis
# ==========================
def get_sentiment(text):
    return analyzer.polarity_scores(text)['compound'] if text.strip() else 0

post_sentiment = get_sentiment(posts)
comment_sentiment = get_sentiment(comments)
status_sentiment = get_sentiment(status)

# ==========================
# ✅ Feature Engineering
# ==========================
avg_sentiment = np.mean([post_sentiment, comment_sentiment, status_sentiment])
sentiment_var = np.var([post_sentiment, comment_sentiment, status_sentiment])
mood_stress_ratio = mood / (stress + 1e-5)
interaction_post_ratio = interaction_freq / (social_posts + 1e-5)
active_hours = activity_level / 2000
screen_pct = screen_time / 24
high_stress = int(stress > 7)
excessive_screen = int(screen_time > 6)
mobile_user = int(location_changes > 3)
sleep_deprivation = int(sleep_hours < 3)
mood_sent_interaction = mood * avg_sentiment
engagement_score = social_posts + interaction_freq + screen_time
phys_risk_score = int((heart_rate > 85) or (heart_rate < 60)) + \
                  int((sleep_hours > 9) or (sleep_hours < 3)) + \
                  int(activity_level < 700) + \
                  int(body_temp > 97)
routine_disruption = np.var([location_changes, calendar_events])

# Placeholder scores (scaler expects them)
anxiety_score = 0
depression_score = 0

# ==========================
# ✅ Prepare DataFrame
# ==========================
user_input = {
    'HeartRate': heart_rate,
    'SleepHours': sleep_hours,
    'ActivityLevel': activity_level,
    'BodyTemperature': body_temp,
    'SocialPosts': social_posts,
    'InteractionFrequency': interaction_freq,
    'ScreenTime': screen_time,
    'LocationChanges': location_changes,
    'CalendarEvents': calendar_events,
    'Mood': mood,
    'StressLevel': stress,
    'DailyReflections': daily_reflections,
    'PostSentiment': post_sentiment,
    'CommentSentiment': comment_sentiment,
    'StatusSentiment': status_sentiment,
    'AverageSentiment': avg_sentiment,
    'SentimentVariance': sentiment_var,
    'MoodStressRatio': mood_stress_ratio,
    'InteractionPostRatio': interaction_post_ratio,
    'ActiveHours': active_hours,
    'ScreenTimePercentage': screen_pct,
    'HighStress': high_stress,
    'ExcessiveScreenTime': excessive_screen,
    'MobileUser': mobile_user,
    'SleepDeprivation': sleep_deprivation,
    'MoodStressDifference': mood - stress,
    'EngagementScore': engagement_score,
    'MoodSentimentInteraction': mood_sent_interaction,
    'PhysicalRiskScore': phys_risk_score,
    'RoutineDisruptionScore': routine_disruption,
    'AnxietyScore': anxiety_score,
    'DepressionScore': depression_score
}

# Convert to DataFrame
X_user = pd.DataFrame([user_input])

# ✅ Apply scaler on full set of original columns
X_user_scaled_all = pd.DataFrame(scaler.transform(X_user[scaled_columns]), columns=scaled_columns)

# ✅ Now select only model-required features after scaling
X_model = X_user_scaled_all[model_feature_columns]

# ==========================
# ✅ Predictions
# ==========================
anx_score = reg_anxiety.predict(X_model)[0]
dep_score = reg_depression.predict(X_model)[0]
anx_lvl = enc_anx.inverse_transform(clf_anxiety.predict(X_model))[0]
dep_lvl = enc_dep.inverse_transform(clf_depression.predict(X_model))[0]
high_stress_pred = clf_stress.predict(X_model)[0]

# ==========================
# ✅ Output Display
# ==========================
st.subheader("📊 Predicted Mental Health Insights")
colA, colB, colC = st.columns(3)
colA.metric("Anxiety Score", f"{anx_score * 100:.2f} / 100")
colB.metric("Depression Score", f"{dep_score * 100:.2f} / 100")
colC.metric("High Stress", "Yes" if high_stress_pred == 1 else "No")
col1, col2 = st.columns(2)
col1.success(f"Predicted Anxiety Level: **{anx_lvl}**")
col2.warning(f"Predicted Depression Level: **{dep_lvl}**")
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and AI models")
