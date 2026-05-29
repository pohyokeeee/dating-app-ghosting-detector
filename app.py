import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 🏢 Page Configuration Profile Architecture Setup
st.set_page_config(
    page_title="Ghosting Analytics Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Custom High-Class Premium Dark UI Styling Injection
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #3b82f6;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        width: 100% !important;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.6) !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_flaml_model():
    model_path = "flaml_best_model.pkl"
    with open(model_path, 'rb') as f:
        return pickle.load(f)

with st.sidebar:
    st.markdown("<h2 style='color: #3b82f6; margin-bottom: 0;'>⚙️ System Kernel</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 13px;'>ML Assignment Production Build v2.1</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        model = load_flaml_model()
        st.markdown("""
            <div style='background-color: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; padding: 12px; border-radius: 10px; text-align: center;'>
                <span style='color: #10b981; font-weight: bold;'>● FLAML ENGINE ONLINE</span>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        # 🚨 THE PERMANENT FIX APPLIED TO SYSTEM BANNER
        st.markdown("""
            <div style='background-color: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; padding: 12px; border-radius: 10px; text-align: center;'>
                <span style='color: #ef4444; font-weight: bold;'>● PIPELINE OFFLINE</span>
            </div>
        """, unsafe_allow_html=True)
        st.stop()
        
    st.markdown("---")
    st.markdown("<p style='color: #64748b; font-size: 12px;'>Faculty of Computer Science and Information Technology<br><b>Universiti Malaya</b></p>", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 40px; margin-bottom: 5px;'>🎯 Predictive Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 16px; margin-bottom: 30px;'>Leveraging specialized Lightweight AutoML optimizations to analyze conversational drop-off risks and relational retention metrics.</p>", unsafe_allow_html=True)

form_container = st.container()
with form_container:
    tab1, tab2 = st.tabs(["👤 Demographic & Profile Architecture", "📈 In-App Behavioral Vectors"])
    
    with tab1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<h3 style='color: #3b82f6; font-size: 18px;'>Identity Profiling</h3>", unsafe_allow_html=True)
            gender_ui = st.selectbox("Gender Identity Spectrum", options=["Female", "Genderfluid", "Male", "Non-binary", "Prefer Not to Say", "Transgender"])
            sexual_orientation = st.selectbox("Orientation Vector", options=[0, 1, 2], format_func=lambda x: ["Bisexual", "Gay", "Pansexual"][x])
            location_type = st.selectbox("Geographic Profile Matrix", options=[0, 1, 2], format_func=lambda x: ["Metropolitan Cluster", "Suburban Matrix", "Urban Core"][x])
        with c2:
            st.markdown("<h3 style='color: #3b82f6; font-size: 18px;'>Account Characteristics</h3>", unsafe_allow_html=True)
            income_bracket = st.selectbox("Estimated Income Segment", options=[0, 1, 2, 3, 4], format_func=lambda x: ["Tier 5 (Very Low)", "Tier 4 (Low)", "Tier 3 (Middle)", "Tier 2 (Upper-Middle)", "Tier 1 (High)"][x])
            profile_pics_count = st.slider("Verified Portrait Display Count", min_value=1, max_value=6, value=3)
            bio_length = st.number_input("Profile Biography Metrics (Character Count)", min_value=0, max_value=1000, value=150)

    with tab2:
        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.markdown("<h3 style='color: #6366f1; font-size: 18px;'>Activity Quantization</h3>", unsafe_allow_html=True)
            app_usage_time_min = st.number_input("Daily Application Retention Window (Minutes)", min_value=0, max_value=1440, value=45)
            app_usage_time_label = st.selectbox("User Retention Classification Index", options=[0, 1, 2], format_func=lambda x: ["Standard Active User", "Elevated Activity Profile", "High-Velocity Power User"][x])
            swipe_right_ratio = st.slider("Swipe-Right Selectivity Balance Rate", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
            swipe_right_label = st.selectbox("Selection Bias Profile", options=[0, 1], format_func=lambda x: ["Highly Selective Sifting", "Optimistic Batch Approvals"][x])
        with c4:
            st.markdown("<h3 style='color: #6366f1; font-size: 18px;'>Interaction Analytics</h3>", unsafe_allow_html=True)
            likes_received = st.number_input("Aggregated Inbound Signals Received", min_value=0, max_value=100000, value=25)
            mutual_matches = st.number_input("Bi-Directional Matches Confirmed", min_value=0, max_value=5000, value=4)
            message_sent_count = st.number_input("Outbound Structural Messaging Volume", min_value=0, max_value=10000, value=18)
            emoji_usage_rate = st.slider("Textual Syntactical Emoji Frequency", min_value=0.0, max_value=1.0, value=0.22, step=0.01)
            last_active_hour = st.slider("Terminal Activity Clock Interaction Hour (0-23)", min_value=0, max_value=23, value=21)
            swipe_time_of_day = st.selectbox("Peak Engagement Access Window", options=[0, 1, 2], format_func=lambda x: ["Late Night Routines", "Early Dawn Browsing", "Mid-Day/Afternoon Blocks"][x])

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 INITIATE GHOSTING MATRIX ANALYSIS"):
    g_fluid = 1 if gender_ui == "Genderfluid" else 0
    g_male = 1 if gender_ui == "Male" else 0
    g_nonbinary = 1 if gender_ui == "Non-binary" else 0
    g_prefer_not = 1 if gender_ui == "Prefer Not to Say" else 0
    g_trans = 1 if gender_ui == "Transgender" else 0
    
    msg_to_usage_ratio = message_sent_count / (app_usage_time_min + 1)
    likes_to_match_ratio = likes_received / (mutual_matches + 1)
    selectivity_index = likes_received / (swipe_right_ratio + 1e-6)
    engagement_score_calc = (message_sent_count + likes_received) / 2.0
    app_usage_time_label_calc = 1 if app_usage_time_min > 30 else 0
    
    input_payload = {
        'gender_genderfluid': g_fluid, 'gender_male': g_male, 'gender_non-binary': g_nonbinary, 
        'gender_prefer not to say': g_prefer_not, 'gender_transgender': g_trans,
        'sexual_orientation': sexual_orientation, 'location_type': location_type, 'income_bracket': income_bracket,
        'app_usage_time_min': app_usage_time_min, 'app_usage_time_label': app_usage_time_label,
        'swipe_right_ratio': swipe_right_ratio, 'swipe_right_label': swipe_right_label,
        'likes_received': likes_received, 'mutual_matches': mutual_matches, 'profile_pics_count': profile_pics_count,
        'bio_length': bio_length, 'message_sent_count': message_sent_count, 'emoji_usage_rate': emoji_usage_rate,
        'last_active_hour': last_active_hour, 'swipe_time_of_day': swipe_time_of_day,
        'selectivity_index': selectivity_index, 'msg_to_usage_ratio': msg_to_usage_ratio,
        'likes_to_match_ratio': likes_to_match_ratio, 'engagement_score_calc': engagement_score_calc,
        'app_usage_time_label_calc': app_usage_time_label_calc
    }
    
    df_inference = pd.DataFrame([input_payload])
    expected_features = [
        'gender_genderfluid', 'gender_male', 'gender_non-binary', 'gender_prefer not to say', 'gender_transgender',
        'sexual_orientation', 'location_type', 'income_bracket', 'app_usage_time_min', 'app_usage_time_label',
        'swipe_right_ratio', 'swipe_right_label', 'likes_received', 'mutual_matches', 'profile_pics_count',
        'bio_length', 'message_sent_count', 'emoji_usage_rate', 'last_active_hour', 'swipe_time_of_day',
        'selectivity_index', 'msg_to_usage_ratio', 'likes_to_match_ratio', 'engagement_score_calc', 'app_usage_time_label_calc'
    ]
    
    if hasattr(model, 'feature_names_in_'):
        features_to_use = list(model.feature_names_in_)
    elif hasattr(model, 'steps') and hasattr(model.steps[0][1], 'feature_names_in_'):
        features_to_use = list(model.steps[0][1].feature_names_in_)
    else:
        features_to_use = [f for f in expected_features if f in df_inference.columns]

    df_inference = df_inference.reindex(columns=features_to_use, fill_value=0.0)
    
    try:
        prediction = model.predict(df_inference)[0]
        probabilities = model.predict_proba(df_inference)[0]
        ghost_prob = probabilities[1]
        
        st.markdown("<h2 style='margin-top: 30px;'>📊 High-Class Analytical Report</h2>", unsafe_allow_html=True)
        rc1, rc2 = st.columns([2, 1], gap="medium")
        
        with rc1:
            if prediction == 1:
                st.markdown(f"""
                    <div style='background: rgba(239, 68, 68, 0.1); border-left: 6px solid #ef4444; padding: 24px; border-radius: 12px;'>
                        <h4 style='color: #f87171; margin-top:0;'>🚨 CRITICAL RISK TRAJECTORY IDENTIFIED</h4>
                        <p style='color: #fca5a5; margin-bottom:0; font-size: 15px;'>The AutoML predictive architecture has flagged this interaction profile as highly volatile. Behavioral indicators match communication drop-off signature profiles with high systemic confidence.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background: rgba(16, 185, 129, 0.1); border-left: 6px solid #10b981; padding: 24px; border-radius: 12px;'>
                        <h4 style='color: #34d399; margin-top:0;'>💚 HEALTHY RETENTION PROFILE MAINTAINED</h4>
                        <p style='color: #a7f3d0; margin-bottom:0; font-size: 15px;'>The interaction metrics present robust operational equilibrium. Messaging frequencies and daily active durations strongly correlate with stable engagement pipelines.</p>
                    </div>
                """, unsafe_allow_html=True)
                
            with st.expander("🔍 Introspect System Engineering Layer"):
                st.markdown(f"""
                    <ul>
                        <li><b>Calculated Engagement Score Metric:</b> {engagement_score_calc:.2f}</li>
                        <li><b>User App Selectivity Factor:</b> {selectivity_index:.4f}</li>
                        <li><b>Outbound Message to Usage Ratio:</b> {msg_to_usage_ratio:.4f}</li>
                    </ul>
                """, unsafe_allow_html=True)
                
        with rc2:
            border_color = "#ef4444" if prediction == 1 else "#10b981"
            st.markdown(f"""
                <div class='metric-card' style='border-top: 5px solid {border_color};'>
                    <p style='text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 5px; font-size: 12px;'>Ghosting Confidence Probability</p>
                    <h2 style='font-size: 48px; color: white; margin: 0;'>{ghost_prob:.2%}</h2>
                    <div style='background-color: #334155; height: 6px; border-radius: 3px; margin-top: 15px; overflow: hidden;'>
                        <div style='background-color: {border_color}; width: {ghost_prob*100}%; height: 100%;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as prediction_error:
        st.error("🚨 In-App Execution Pipeline Assessment Error")
        st.code(str(prediction_error))