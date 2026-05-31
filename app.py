import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import os
import streamlit.components.v1 as components
import textwrap


# -----------------------------
# LOAD FULL PIPELINE
# -----------------------------
PIPELINE_PATH = "flaml_best_model.pkl"
pipeline = joblib.load(PIPELINE_PATH)


# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Ghosted Predictor",
    page_icon="💔",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>

/* Hide hamburger menu */
#MainMenu {
    visibility: hidden;
}

/* Hide footer */
footer {
    visibility: hidden;
}

/* Hide header */
header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------
# SESSION STATE
# ---------------------------------
if 'nav_index' not in st.session_state:
    st.session_state.nav_index = 0

if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

def set_nav_page(index):
    st.session_state.nav_index = index
    st.rerun()

# ---------------------------------
# CUSTOM CSS
# ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Syne:wght@400;600;700&display=swap');

html, body, [class*='css'] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #070B1A;
    color: white;
}

.stApp {
    background:
        radial-gradient(circle at 20% 50%, rgba(255,0,110,0.15), transparent 30%),
        radial-gradient(circle at 80% 80%, rgba(131,56,236,0.15), transparent 30%),
        radial-gradient(circle at 40% 20%, rgba(58,134,255,0.1), transparent 30%),
        #070B1A;
}

.hero-title{
    font-family: 'Syne', sans-serif;
    font-size: 85px;
    line-height: 1.0;
    font-weight: 700;
    margin-bottom: 20px;
    white-space: nowrap;
}

.hero-sub{e
    color:#00ffcc;
    font-size:28px;
    font-weight:700;
}

.hero-desc{
    color:#bbbbcc;
    font-size:20px;
    margin-top:25px;
    max-width:700px;
    line-height:1.7;
}

.stat-card{
    background: rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:20px;
    padding:30px;
    text-align:center;
    backdrop-filter: blur(12px);
    transition:0.3s;
}

.stat-card:hover{
    transform:translateY(-6px);
    border:1px solid #ff0080;
}

.stat-num{
    font-size:42px;
    font-weight:700;
    color:#ff0080;
}

.stat-label{
    color:#bbbbcc;
    margin-top:10px;
    font-size:18px;
}

.section-title {
    font-size:50px;
    font-weight:700;
    margin-bottom:30px;
    font-family: 'Syne', sans-serif;
}

.stButton>button {
    width:100%;
    background:linear-gradient(90deg,#ff0080,#8b5cf6);
    color:white;
    border:none;
    border-radius:15px;
    padding:18px;
    font-size:22px;
    font-weight:700;
    box-shadow:0px 0px 25px rgba(255,0,128,0.5);
    transition:0.3s;
}

.stButton>button:hover {
    transform:scale(1.02);
}

.result-box {
    background: rgba(255,255,255,0.04);
    padding:40px;
    border-radius:25px;
    border:1px solid rgba(255,255,255,0.1);
    margin-top:20px;
    text-align: center;
}

.heart {
    position: relative;
    width: 220px;
    height: 220px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 0, 128, 0.15) 100%);
    backdrop-filter: blur(25px);
    transform: rotate(-45deg);
    animation: heartbeat 1.5s infinite ease-in-out;
}

.heart::before,
.heart::after {
    content: '';
    position: absolute;
    width: 220px;
    height: 220px;
    background: inherit;
    border-radius: 50%;
}

.heart::before {
    top: -110px;
    left: 0;
}

.heart::after {
    top: 0;
    left: 110px;
}

@keyframes heartbeat {
    0% {
        transform: rotate(-45deg) scale(0.9);
    }

    50% {
        transform: rotate(-45deg) scale(1.02);
        box-shadow: 0 0 120px rgba(255, 0, 128, 0.6);
    }

    100% {
        transform: rotate(-45deg) scale(0.9);
    }
}

.built-dev-badge{
    display:inline-block;
    padding:8px 12px;
    border-radius:12px;
    background: linear-gradient(90deg, rgba(255,0,128,0.22), rgba(139,92,246,0.22));
    border: 1px solid rgba(255,255,255,0.14);
    color: white;
    font-size:14px;
    font-weight:700;
}

.home-dev-corner{
    position: fixed;
    top: 20px;
    right: 24px;
    z-index: 9999;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# NAVBAR
# ---------------------------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Predict", "Models", "Insights", "About"],
    icons=["house", "cpu", "bar-chart", "graph-up", "info-circle"],
    orientation="horizontal",
    manual_select=st.session_state.nav_index,
    key="main_nav",
    styles={
        "container": {
            "background-color": "#070B1A",
            "padding": "15px"
        },
        "nav-link-selected": {
            "background": "linear-gradient(90deg,#ff0080,#8b5cf6)"
        }
    }
)

# ---------------------------------
# HOME PAGE
# ---------------------------------
if selected == "Home":

    st.session_state.prediction_done = False

    st.markdown(
        "<span class='built-dev-badge home-dev-corner'>Developed by AR. Shezan</span>",
        unsafe_allow_html=True
    )

    left, right = st.columns([1.5,1], gap="large")

    with left:

        st.markdown(
            "<div class='hero-title'>Ghosted Predictor</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='hero-sub'>Dating Analytics Platform</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='hero-desc'>Discover your relationship destiny using futuristic machine learning. Predict ghosting probability, analyze dating behavior and gain deep relationship insights.</div>",
            unsafe_allow_html=True
        )

        st.write(" ")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                "<div class='stat-card'><div class='stat-num'>50K+</div><div class='stat-label'>Records Analyzed</div></div>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                "<div class='stat-card'><div class='stat-num'>6</div><div class='stat-label'>ML Model</div></div>",
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                "<div class='stat-card'><div class='stat-num'>87.75%</div><div class='stat-label'>KNN Accuracy</div></div>",
                unsafe_allow_html=True
            )

        st.write(" ")

        if st.button("✨ Start Prediction Now .... !"):
            set_nav_page(1)

    with right:
        st.markdown(
            "<div style='display:flex; justify-content:center; align-items:center; height:500px;'><div class='heart'></div></div>",
            unsafe_allow_html=True
        )

# ---------------------------------
# PREDICT PAGE
# ---------------------------------
elif selected == "Predict":

    st.markdown("""
    <style>

    .predict-title{
        font-size:58px;
        font-weight:800;
        margin-top:10px;
        margin-bottom:35px;
        font-family:'Syne', sans-serif;
        color:white;
    }

    .sub-title{
        font-size:22px;
        font-weight:700;
        margin-bottom:25px;
        color:white;
    }

    .stSelectbox label,
    .stSlider label,
    .stNumberInput label{
        color:white !important;
        font-size:18px !important;
        font-weight:500;
    }

    div[data-baseweb="select"] > div{
        background-color:#1f2230 !important;
        border-radius:12px !important;
        border:none !important;
    }

    div[data-baseweb="input"] > div{
        background-color:#1f2230 !important;
        border-radius:12px !important;
        border:none !important;
    }

    .stSlider > div > div > div > div{
        background:#ff4d67 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='predict-title'>Ghosting Prediction</div>",
        unsafe_allow_html=True
    )

    container = st.container(border=True)

    with container:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                "<div class='sub-title'>👤 Demographics</div>",
                unsafe_allow_html=True
            )

            gender = st.selectbox(
                "Gender Orientation",
                ["Male", "Female", "Non-Binary"]
            )

            sexual = st.selectbox(
                "Sexual Orientation",
                ["Straight", "Gay", "Bisexual"]
            )

            region = st.selectbox(
                "Regional Profiling",
                ["Urban", "Suburban", "Rural"]
            )

            income = st.selectbox(
                "Annual Income Tier",
                ["<30k", "30k-60k", "60k-100k", "100k+"]
            )

        with col2:

            st.markdown(
                "<div class='sub-title'>🧮 Activity Workspace</div>",
                unsafe_allow_html=True
            )

            education = st.selectbox(
                "Completed Education",
                ["HS", "Bachelor", "Master", "PhD"]
            )

            usage = st.number_input(
                "Daily Application Time (mins)",
                min_value=0,
                max_value=500,
                value=95
            )

            likes = st.number_input(
                "Cumulative Likes Received",
                min_value=0,
                max_value=10000,
                value=120
            )

            matches = st.number_input(
                "Mutual Matches Achieved",
                min_value=0,
                max_value=1000,
                value=15
            )

        with col3:

            st.markdown(
                "<div classt='sub-title'>💬 Behavioral Matrix</div>",
                unsafe_allow_html=True
            )

            swipe = st.slider(
                "Swipe-Right Approval Rate",
                0.0,
                1.0,
                0.42
            )

            messages = st.number_input(
                "Sent Message Volume",
                min_value=0,
                max_value=10000,
                value=30
            )

            bio = st.slider(
                "Bio Word Count Profile",
                0,
                500,
                150
            )

            photos = st.slider(
                "Uploaded Pictures",
                0,
                10,
                4
            )

        st.write(" ")

        if st.button("Analyze Interaction Vector 🚀"):
            st.session_state.prediction_done = True

        # -------------------------
    # RESULTS
    # -------------------------
    if st.session_state.prediction_done:

        components.html(
            """
            <script>
            setTimeout(function() {
                const element = document.getElementById('prediction-results');
                if (element) {
                    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 300);
            </script>
            """,
            height=0
        )

        # Removed unused variables
        # gender, education, spam_score, is_empty_profile, engagement_score

        selectivity_index = likes / (swipe + 1e-6)
        # app_usage_norm = usage / 500 # This variable is not used in the new input_data creation.
        # engagement_score = app_usage_norm * 1 # Removed as per user instruction.
        msg_to_usage_ratio = messages / (usage + 1)
        likes_to_match_ratio = likes / (matches + 1)
        # spam_score = messages * swipe # Removed as per user instruction.
        # is_empty_profile = int((bio < 10) and (photos == 1)) # Removed as per user instruction.

        # New calculations for engagement_score_calc and app_usage_time_label_calc
        engagement_score_calc = usage / 500
        app_usage_time_label_calc = 0

        input_data = pd.DataFrame([
            {
                'sexual_orientation': {"Straight": 3, "Gay": 1, "Bisexual": 0}[sexual],
                'location_type': {"Urban": 5, "Suburban": 4, "Rural": 0}[region],
                'income_bracket': {"<30k": 0, "30k-60k": 1, "60k-100k": 3, "100k+": 6}[income],
                'app_usage_time_min': usage,
                'app_usage_time_label': 0,
                'swipe_right_ratio': swipe,
                'swipe_right_label': 0,
                'likes_received': likes,
                'mutual_matches': matches,
                'profile_pics_count': photos,
                'bio_length': bio,
                'message_sent_count': messages,
                'emoji_usage_rate': 0.0,
                'last_active_hour': 12,
                'swipe_time_of_day': 0,
                'selectivity_index': selectivity_index,
                'msg_to_usage_ratio': msg_to_usage_ratio,
                'likes_to_match_ratio': likes_to_match_ratio,
                'engagement_score_calc': engagement_score_calc,
                'app_usage_time_label_calc': app_usage_time_label_calc
            }
        ])

        input_data = input_data[[
            'sexual_orientation',
            'location_type',
            'income_bracket',
            'app_usage_time_min',
            'app_usage_time_label',
            'swipe_right_ratio',
            'swipe_right_label',
            'likes_received',
            'mutual_matches',
            'profile_pics_count',
            'bio_length',
            'message_sent_count',
            'emoji_usage_rate',
            'last_active_hour',
            'swipe_time_of_day',
            'selectivity_index',
            'msg_to_usage_ratio',
            'likes_to_match_ratio',
            'engagement_score_calc',
            'app_usage_time_label_calc'
        ]]

        prediction_probability = pipeline.predict_proba(input_data)[0][1]
        risk = int(prediction_probability * 100)

        status = (
            "🚩 High Risk"
            if risk > 70
            else "⚠️ Moderate Risk"
            if risk > 40
            else "✅ Low Risk"
        )

        color = (
            "#ff0040"
            if risk > 70
            else "#ffaa00"
            if risk > 40
            else "#00ffcc"
        )

        st.markdown(
            f"""
            <div class='result-box' id='prediction-results'>
                <h1 style='color:{color};'>{risk}%</h1>
                <h2>{status}</h2>
                <p style='font-size:20px;color:#bbbbcc'>
                AI detected your ghosting probability based on dating behavior.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk,
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff0080"}
                }
            )
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# MODELS PAGE
# ---------------------------------
elif selected == "Models":

    st.markdown("""
    <style>

    .model-title{
        font-size:64px;
        font-weight:800;
        margin-top:10px;
        margin-bottom:35px;
        color:white;
    }

    .best-model-box{
        background:rgba(0,255,120,0.15);
        border:1px solid rgba(0,255,120,0.2);
        padding:22px;
        border-radius:16px;
        margin-top:25px;
        font-size:22px;
        color:#4dff88;
        font-weight:600;
    }

    .table-wrapper{
        border-radius:20px;
        overflow:hidden;
        border:1px solid rgba(255,255,255,0.08);
    }

    div[data-testid="stDataFrame"]{
        border:none !important;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='model-title'>
    🎧 Model Selection & Comparison
    </div>
    """, unsafe_allow_html=True)

    model_df = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost",
        "Logistic Regression",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "AutoML Champion (XGBoost)"
    ],
    "Accuracy": [0.8361, 0.7060, 0.4910, 0.4908, 0.8775, 0.8780],
    "Macro F1 Score": [0.4999, 0.4798, 0.3969, 0.3969, 0.4936, 0.4894]
   }, index=[1, 2, 3, 4, 5, 6])

    model_df.index.name = "No."

    st.dataframe(model_df, use_container_width=True)


    st.markdown("""
<div class='best-model-box'>
🏆 Best Performing Model: <b>Random Forest</b>
(Accuracy: <b>83.61%</b>, Macro F1: <b>49.99%</b>)
</div>
""", unsafe_allow_html=True)

    st.write(" ")

    fig = px.bar(
        model_df,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        color="Accuracy",
        color_continuous_scale="purples"
    )

    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#070B1A",
        font_color="white",
        height=500,
        title="Model Accuracy Comparison",
        title_font_size=28,
        xaxis_title="",
        yaxis_title="Accuracy",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# INSIGHTS PAGE
# ---------------------------------
elif selected == "Insights":

    st.markdown(
        "<div class='section-title'>Insights Dashboard</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "<div class='stat-card'><div class='stat-num'>9.92%</div><div class='stat-label'>Ghosted Profiles</div></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            "<div class='stat-card'><div class='stat-num'>90.08%</div><div class='stat-label'>Successful Interactions</div></div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            "<div class='stat-card'><div class='stat-num'>50K</div><div class='stat-label'>Dataset Records</div></div>",
            unsafe_allow_html=True
        )

    pie = px.pie(
        names=["Ghosted","Healthy"],
        values=[9.92,90.08],
        hole=0.5,
        color_discrete_sequence=["#ff0080","#8b5cf6"]
    )

    pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=450
    )

    st.plotly_chart(pie, use_container_width=True)

# ---------------------------------
# ABOUT PAGE
# ---------------------------------
elif selected == "About":

    st.markdown(
        "<div class='section-title'>About Ghosted Predictor</div>",
        unsafe_allow_html=True
    )

    st.markdown(
                textwrap.dedent("""
<div class='result-box'>
    <h2>Dating Analytics Platform</h2>
    <p style='font-size:20px;color:#bbbbcc; line-height:1.8'>
        Ghosted Predictor is an intelligent machine learning platform designed to analyze dating app interactions and estimate the likelihood of ghosting.
        By leveraging behavioral patterns, user engagement metrics and communication trends, the system provides data-driven relationship insights.
        The platform evaluates multiple predictive models to identify the most effective approach for ghosting detection.
        Through interactive analytics and visualization, users can better understand their dating behaviors and communication outcomes.
        Our goal is to transform complex dating data into meaningful predictions that support informed relationship decisions.....
    </p>
                         
    ""
                                
        UI built & developed by : AR. Shezan (arshezan.my@gmail.com)
</div>
                """),

                  unsafe_allow_html=True
                  )

else:

    st.session_state.prediction_done = False
    st.write(f"Welcome to the {selected} page.")
