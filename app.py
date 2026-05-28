import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set professional layout configurations
st.set_page_config(page_title="Ghosting Prediction Engine", layout="wide")

st.title("🎯 Dating App Ghosting Predictive Dashboard")
st.markdown("Enter user behavior and interaction analytics below to evaluate the probability of a communication drop-off (Ghosting).")
st.markdown("---")

# Cached loader to ensure the app stays lightning-fast
@st.cache_resource
def load_flaml_model():
    model_path = "flaml_best_model.pkl"
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Load the engine
try:
    model = load_flaml_model()
    st.sidebar.success("✅ FLAML AutoML Engine Active")
except Exception as e:
    st.sidebar.error("❌ Model Loading Error")
    st.sidebar.write(str(e))
    st.stop()  # Stop app execution if model can't be fetched

# ==============================================================================
# 1. INTERFACE INPUTS: Setup Columns for User Attributes & Engagement Metrics
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 User Profile Metrics")
    age = st.slider("User Age", min_value=18, max_value=70, value=25)
    gender_encoded = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    location_distance_km = st.number_input("Location Distance Between Matches (km)", min_value=0.0, max_value=500.0, value=15.0, step=0.5)
    profile_completion_percentage = st.slider("Profile Completion %", min_value=0, max_value=100, value=85)
    
    # NEW: Move missing features onto the UI to unlock your model's branches!
    bio_length = st.slider("Profile Bio Character Length", min_value=0, max_value=500, value=120)
    education_level = st.selectbox("Education Level", options=[0, 1, 2, 3], format_func=lambda x: ["High School", "Undergraduate", "Postgraduate", "Other"][x])

with col2:
    st.subheader("📈 In-App Engagement & Activity Logs")
    app_usage_time_min = st.number_input("Daily App Usage Duration (Minutes)", min_value=0, max_value=1440, value=45)
    swipes_per_day = st.number_input("Swipes Registered Per Day", min_value=0, max_value=5000, value=60)
    message_sent_count = st.number_input("Messages Sent", min_value=0, max_value=10000, value=12)
    message_received_count = st.number_input("Messages Received", min_value=0, max_value=10000, value=10)
    likes_received = st.number_input("Total Likes Received", min_value=0, max_value=100000, value=25)
    mutual_matches = st.number_input("Mutual Matches Sparked", min_value=0, max_value=1000, value=3)

st.markdown("---")

# ==============================================================================
# 2. PIPELINE INFERENCE ENGINE (Strict Type & Alignment Enforcement)
# ==============================================================================
if st.button("🚀 Analyze Ghosting Risk Profile", type="primary"):
    
    # 1. Dynamically compute ratio metrics as explicit floats
    msg_to_usage_ratio = float(message_sent_count) / (float(app_usage_time_min) + 1.0)
    likes_to_match_ratio = float(likes_received) / (float(mutual_matches) + 1.0)
    
    # 2. Compute active behavioral index features as explicit floats
    engagement_score = (float(message_sent_count) + float(swipes_per_day)) / 2.0
    selectivity_index = float(mutual_matches) / (float(swipes_per_day) + 1.0)
    app_usage_time_label = 1.0 if app_usage_time_min > 30 else 0.0
    
    # 3. Map to input schema, casting every item to float
    input_data = {
        'age': float(age),
        'gender_encoded': float(gender_encoded),
        'location_distance_km': float(location_distance_km),
        'app_usage_time_min': float(app_usage_time_min),
        'swipes_per_day': float(swipes_per_day),
        'message_sent_count': float(message_sent_count),
        'message_received_count': float(message_received_count),
        'profile_completion_percentage': float(profile_completion_percentage),
        'likes_received': float(likes_received),
        'mutual_matches': float(mutual_matches),
        'msg_to_usage_ratio': msg_to_usage_ratio,
        'likes_to_match_ratio': likes_to_match_ratio,
        'Engagement_Score': engagement_score,
        'Selectivity_Index': selectivity_index,
        'app_usage_time_label': app_usage_time_label,
        'bio_length': float(bio_length),
        'education_level': float(education_level)
    }
    
    # 4. Form DataFrame and strictly enforce 64-bit float representation
    df_inference = pd.DataFrame([input_data]).astype(np.float64)
    
    # 5. Lock column placement to match training logs
    try:
        expected_features = model.feature_names_in_
        df_inference = df_inference.reindex(columns=expected_features, fill_value=0.0)
    except AttributeError:
        pass
    
    try:
        # Run calculation
        prediction = model.predict(df_inference)[0]
        probabilities = model.predict_proba(df_inference)[0]
        
        st.subheader("📊 Analytical Diagnostic Results")
        c1, c2 = st.columns(2)
        with c1:
            if prediction == 1:
                st.error("🚨 **High Risk Warning:** User pattern reflects signature **Is_Ghosted** behaviors.")
            else:
                st.success("💚 **Stable Status:** User interaction levels track as **Not_Ghosted** profile.")
                
        with c2:
            st.metric(label="Calculated Ghosting Confidence Rate", value=f"{probabilities[1]:.2%}")
            
    except Exception as prediction_error:
        st.error("🚨 Pipeline Assessment Error")
        st.write(str(prediction_error))