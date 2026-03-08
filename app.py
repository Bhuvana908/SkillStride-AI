import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- Page Configuration ---
st.set_page_config(page_title="SkillStride AI | Career Intelligence", layout="wide")

# --- HIGH-END UI STYLING ---
st.markdown("""
    <style>
    /* 1. COMPLETELY ELIMINATE TOP WHITESPACE */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 0rem !important;
        margin-top: -60px !important; /* Pulls content into the header area */
    }

    /* 2. BACKGROUND & BASE THEME */
    .stApp {
        background: #0f172a;
        color: #f8fafc;
    }
    
    /* 3. SIDEBAR VISIBILITY FIXES */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* High contrast labels */
    [data-testid="stSidebar"] label p {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }

    /* Multiselect Tags Visibility */
    [data-baseweb="tag"] {
        background-color: #3b82f6 !important;
    }
    [data-baseweb="tag"] span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* 4. PROFESSIONAL CARDS & TEXT */
    .job-card {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.3s ease;
    }
    .job-card:hover {
        border-color: #38bdf8;
        transform: translateY(-8px);
        background: rgba(255, 255, 255, 0.07);
    }

    .job-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .match-percent {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Empty state text style */
    .empty-text {
        color: #94a3b8;
        font-style: italic;
        font-size: 0.95rem;
        padding: 10px 0;
    }

    /* 5. BUTTON STYLING */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-weight: 700;
        transition: 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Artifacts ---
@st.cache_resource
def load_data():
    with open('skill_gap_artifacts.pkl', 'rb') as f:
        data = pickle.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return data, model

artifacts, embed_model = load_data()
df_jobs = artifacts['df_jobs']
job_embeddings = artifacts['job_embeddings']
all_skills = artifacts['all_skills']
all_jobs = artifacts['all_job_titles']

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=60)
    st.markdown("<h2 style='color:white; margin-top:0;'>SkillStride AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    user_skills = st.multiselect("🌟 Select Your Expertise:", options=all_skills)
    preferred_job = st.selectbox("🎯 Target Career Role:", options=all_jobs)
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Generate Career Report")

# --- Main Page UI ---
if analyze_btn and user_skills:
    user_set = set(s.lower() for s in user_skills)
    user_text = ", ".join(user_skills)
    user_embedding = embed_model.encode(user_text).reshape(1, -1)
    
    # Header Section
    st.markdown("<h1 style='text-align: center; color: white; padding: 60px 0 40px 0;'>Career Intelligence Dashboard</h1>", unsafe_allow_html=True)
    
    # Section 1: Job Matches
    st.markdown("### 🔍 Immediate Opportunities")
    similarities = cosine_similarity(user_embedding, job_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:3]
    
    cols = st.columns(3)
    for i, idx in enumerate(top_indices):
        job_row = df_jobs.iloc[idx]
        score = int(round(similarities[idx] * 100))
        with cols[i]:
            st.markdown(f"""
                <div class="job-card">
                    <div class="job-title">{job_row['Job Title']}</div>
                    <div class="match-percent">{score}% Match</div>
                </div>
            """, unsafe_allow_html=True)

    # Section 2: Path Analysis
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"### 🚩 Path to {preferred_job}")
    
    target_row = df_jobs[df_jobs['Job Title'] == preferred_job].iloc[0]
    req_skills = set(s.lower() for s in target_row['Required Skills'])
    matched = sorted(list(req_skills & user_set))
    missing = sorted(list(req_skills - user_set))
    
    readiness = len(matched) / len(req_skills)
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.markdown(f"""
            <div class="job-card" style="text-align: left;">
                <h4 style="color: #94a3b8; margin:0;">Readiness Score</h4>
                <div style="font-size: 3.5rem; font-weight: 800; color: #4ade80;">{int(readiness*100)}%</div>
                <p style="opacity: 0.6; font-size: 0.9rem;">Analysis based on {len(req_skills)} core skills.</p>
            </div>
        """, unsafe_allow_html=True)

    with res_col2:
        # Mastered Skills with Empty State
        st.markdown("#### ✅ Mastered Skills")
        if matched:
            tags = "".join([f'<span class="skill-tag" style="background:rgba(74,222,128,0.1); color:#4ade80; padding:5px 12px; border-radius:6px; margin:4px; display:inline-block; border:1px solid rgba(74,222,128,0.3); font-weight:600;">{s.title()}</span>' for s in matched])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown('<p class="empty-text">No matching skills identified for this role yet.</p>', unsafe_allow_html=True)
        
        # Critical Gaps with Empty State
        st.markdown("<br>⚡ Critical Gaps", unsafe_allow_html=True)
        if missing:
            tags = "".join([f'<span class="skill-tag" style="background:rgba(251,113,133,0.1); color:#fb7185; padding:5px 12px; border-radius:6px; margin:4px; display:inline-block; border:1px solid rgba(251,113,133,0.3); font-weight:600;">{s.title()}</span>' for s in missing])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#4ade80; font-weight:bold;">Elite Status: You have mastered all core skills for this role!</p>', unsafe_allow_html=True)

else:
    # Landing / Welcome Page
    st.markdown("""
        <div style="text-align: center; padding: 120px 20px;">
            <h1 style="font-size: 4.5rem; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; margin-bottom: 20px;">
                SkillStride AI
            </h1>
            <p style="font-size: 1.5rem; color: #94a3b8; max-width: 800px; margin: 0 auto; line-height: 1.6;">
                The next generation of career mapping. Identify your expertise, 
                discover market fits, and bridge the gap to your next promotion.
            </p>
            <div style="margin-top: 50px; color: #3b82f6; font-weight: 800; font-size: 1.2rem; letter-spacing: 1px;">
                ← ENTER YOUR PROFILE DETAILS TO BEGIN
            </div>
        </div>
    """, unsafe_allow_html=True)