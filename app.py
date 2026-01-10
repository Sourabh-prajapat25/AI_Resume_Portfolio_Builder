import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="AI Resume & Portfolio Builder",
    page_icon="📄",
    layout="centered"
)

# 🔑 CONFIGURE GEMINI API KEY
genai.configure(api_key="AIzaSyBKwomwbNEb0zJYwKMhMXR0Fw5wfR4TYnc")
gen_model = genai.GenerativeModel("gemini-3-flash-preview")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("📌 About This Project")
st.sidebar.info(
    """
    **AI Resume & Portfolio Builder**

    🔹 NLP + Machine Learning  
    🔹 Resume-to-Job Role Matching  
    🔹 Skill Improvement Suggestions  
    🔹 AI-Generated Resume Summary  

    **Built for:** AICTE–Edunet AIML Internship  
    **Developer:** Sourabh Prajapat
    """
)

# -------------------------------------------------
# LOAD DATA & MODEL
# -------------------------------------------------
df = pd.read_csv("resume_data.csv")

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

X = vectorizer.transform(df["skills"])

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🤖 AI Resume & Portfolio Builder</h1>
    <p style='text-align:center; font-size:18px;'>
    Analyze resume skills, find best job role, and generate AI-powered resume summary
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

resume_input = st.text_area(
    "📝 Enter your resume skills (comma separated):",
    placeholder="e.g. Python, Machine Learning, React, SQL, Data Analysis"
)

# -------------------------------------------------
# ANALYZE RESUME
# -------------------------------------------------
if st.button("🚀 Analyze Resume"):

    if resume_input.strip() == "":
        st.warning("⚠️ Please enter your resume skills.")

    else:
        resume_vec = vectorizer.transform([resume_input])
        similarity = cosine_similarity(resume_vec, X)
        best_match = similarity.argmax()

        role = df.iloc[best_match]["role"]
        score = similarity[0][best_match]

        # RESULTS
        st.success(f"🎯 Best Matching Role: **{role}**")
        st.info(f"📊 Match Score: **{round(score * 100, 2)}%**")

        # Resume Strength Indicator
        if score > 0.7:
            st.success("💪 Resume Strength: Strong Match")
        elif score > 0.4:
            st.warning("🙂 Resume Strength: Average Match – Can Improve")
        else:
            st.error("❌ Resume Strength: Weak Match – Skill Upgrade Needed")

        # Skill Suggestions
        st.subheader("📌 Suggested Skills to Improve")
        st.write("Based on similar successful profiles, consider improving:")
        st.code(df.iloc[best_match]["skills"])

        # -------------------------------------------------
        # GENERATIVE AI FEATURE
        # -------------------------------------------------
        st.subheader("✍️ AI-Generated Resume Summary")

        if st.button("✨ Generate Resume Summary"):
            with st.spinner("Generating professional summary using AI..."):
                prompt = f"""
                Create a professional resume summary for a student.
                Skills: {resume_input}
                Target Job Role: {role}

                The summary should be concise, professional, and placement-ready.
                """

                response = gen_model.generate_content(prompt)
                st.write(response.text)
