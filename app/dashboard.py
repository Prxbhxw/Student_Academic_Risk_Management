import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure current script directory and parent directory are on sys.path for cloud deployment
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

for p in [str(CURRENT_DIR), str(ROOT_DIR), str(ROOT_DIR / "app")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from analysis import analyse, confidence_label, get_recommendations, get_model_metrics
    from charts import CATEGORIES, make_donut, make_academic_chart, make_engagement_chart
    from report import build_pdf
except ImportError:
    from app.analysis import analyse, confidence_label, get_recommendations, get_model_metrics
    from app.charts import CATEGORIES, make_donut, make_academic_chart, make_engagement_chart
    from app.report import build_pdf

st.set_page_config(page_title="Student Academic Risk Predictor", layout="wide", page_icon="🎓")

# 20 Feature columns (exact order used in model training)
feature_cols = [
    'attendance_percentage', 'avg_test_score', 'assignments_submitted_pct',
    'study_hours_per_week', 'previous_semester_gpa', 'extracurricular_participation',
    'backlogs', 'library_hours_per_week', 'achievements', 'num_activities',
    'prev_failed_courses', 'online_learning_hours', 'tutoring_participation',
    'competition_participation', 'internship_experience', 'learning_plan_completion_pct',
    'weekly_self_learning_hours', 'practical_opportunity_freq', 'discussion_participation_freq',
    'exercise_hours_per_week'
]

# ---- Load the dataset with dynamic path resolution ----
DATA_CANDIDATES = [
    ROOT_DIR / "data" / "students_dataset.csv",
    CURRENT_DIR / "data" / "students_dataset.csv",
    CURRENT_DIR.parent.parent / "data" / "students_dataset.csv",
    Path("data/students_dataset.csv"),
    Path("student-risk-prediction-main/data/students_dataset.csv"),
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    st.error("Error: Could not locate 'students_dataset.csv'. Please ensure 'data/students_dataset.csv' exists in your repository.")
    st.stop()

df = pd.read_csv(DATA_PATH)
avg = df[feature_cols].mean()

# ---- Load or train model with cache fallback for cloud deployment ----
MODEL_CANDIDATES = [
    CURRENT_DIR / "risk_model.pkl",
    ROOT_DIR / "app" / "risk_model.pkl",
    Path("app/risk_model.pkl"),
    Path("risk_model.pkl"),
    Path("student-risk-prediction-main/app/risk_model.pkl"),
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)

@st.cache_resource
def load_or_train_model():
    if MODEL_PATH is not None:
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    # Fallback for cloud environments with different python/scikit-learn versions
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    X = df[feature_cols]
    y = df['risk_label']
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=3000, random_state=42))
    ])
    pipe.fit(X, y)
    return pipe

model = load_or_train_model()

ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
CARD_BG = {"Low": "#d4f7dc", "Medium": "#fff3cd", "High": "#f8d7da"}
CARD_TEXT = {"Low": "#1e7e34", "Medium": "#856404", "High": "#a71d2a"}
STATUS_SCREEN = {'pos': "🟢 Positive", 'neu': "🟡 Neutral", 'bad': "🔴 Needs Attention"}

# ---- Header ----
st.title("🎓 Student Academic Risk Prediction")
st.write("A multi-dimensional machine learning early-warning system for identifying students requiring academic support. "
         "Evaluates academic performance, previous course failure history, self-learning habits, online engagement, tutoring, "
         "internships, competition experience, and wellness.")

# ---- Sidebar inputs ----
with st.sidebar:
    st.header("📋 Student Profile Inputs")

    with st.expander("👤 Student Information", expanded=True):
        student_name = st.text_input("Student Name", placeholder="e.g. Alex Morgan")
        student_usn = st.text_input("USN / Student ID", placeholder="e.g. 1MS21CS001")

    with st.expander("📚 Academic Performance & History", expanded=True):
        attendance = st.slider("Attendance Percentage (%)", 0, 100, 75)
        test_score = st.slider("Average Test Score (0-100)", 0, 100, 65)
        assignments = st.slider("Assignments Submitted (%)", 0, 100, 80)
        gpa = st.slider("Previous Semester GPA (0.0 - 10.0)", 0.0, 10.0, 6.5, step=0.1)
        backlogs = st.slider("Number of Active Backlogs", 0, 5, 0)
        prev_failed = st.slider("Previous Semester Failed Courses", 0, 5, 0)

    with st.expander("⏱️ Study Habits & Self-Regulation", expanded=False):
        study_hours = st.slider("Study Hours per Week", 0, 40, 12)
        weekly_self_learning = st.slider("Weekly Self-Learning Time (hrs/wk)", 0, 35, 11)
        online_learning = st.slider("Online Learning Duration (hrs/wk)", 0, 30, 8)
        library_hours = st.slider("Library Hours per Week", 0, 20, 6)
        learning_plan_pct = st.slider("Personal Learning Plan Completion (%)", 0, 100, 75)

    with st.expander("🤝 Tutoring, Engagement & Wellness", expanded=False):
        tutoring = st.selectbox("Academic Tutoring Participation", ["No", "Yes"], index=0)
        tutoring_val = 1 if tutoring == "Yes" else 0
        discussion_freq = st.slider("Academic Discussion Participation Frequency (/wk)", 0, 10, 4)
        exercise_hours = st.slider("Exercise Duration and Frequency (hrs/wk)", 0, 14, 5)

        extracurricular = st.selectbox("Extracurricular Participation", ["No", "Yes"], index=0)
        extracurricular_val = 1 if extracurricular == "Yes" else 0
        if extracurricular == "Yes":
            num_activities = st.slider("Number of Activities/Clubs", 0, 5, 1)
            achievements = st.slider("Achievements / Prizes Won", 0, 5, 0)
        else:
            num_activities = 0
            achievements = 0

    with st.expander("💼 Practical & Competition Experience", expanded=False):
        competition_part = st.slider("Subject-Related Competition Participation", 0, 5, 1)
        internship = st.selectbox("Internship and Practical Experience", ["No", "Yes"], index=0)
        internship_val = 1 if internship == "Yes" else 0
        if internship == "Yes":
            practical_freq = st.slider("Internship & Practical Opportunity Frequency", 0, 5, 1)
        else:
            practical_freq = 0

    predict_clicked = st.button("🚀 Predict Risk Level", type="primary", use_container_width=True)

# ---- Run prediction and store in session_state ----
if predict_clicked:
    values = {
        'attendance_percentage': attendance,
        'avg_test_score': test_score,
        'assignments_submitted_pct': assignments,
        'study_hours_per_week': study_hours,
        'previous_semester_gpa': gpa,
        'extracurricular_participation': extracurricular_val,
        'backlogs': backlogs,
        'library_hours_per_week': library_hours,
        'achievements': achievements,
        'num_activities': num_activities,
        # 10 New features
        'prev_failed_courses': prev_failed,
        'online_learning_hours': online_learning,
        'tutoring_participation': tutoring_val,
        'competition_participation': competition_part,
        'internship_experience': internship_val,
        'learning_plan_completion_pct': learning_plan_pct,
        'weekly_self_learning_hours': weekly_self_learning,
        'practical_opportunity_freq': practical_freq,
        'discussion_participation_freq': discussion_freq,
        'exercise_hours_per_week': exercise_hours,
    }
    input_data = pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    st.session_state['result'] = {
        'name': student_name,
        'usn': student_usn,
        'prediction': prediction,
        'probs': {cls: round(float(p) * 100, 1) for cls, p in zip(model.classes_, proba)},
        'values': values,
        'extracurricular': extracurricular,
    }

if 'result' not in st.session_state:
    st.info("👈 Enter the student's details in the sidebar and click **🚀 Predict Risk Level** to run assessment.")
    st.stop()

# ---- Retrieve results ----
res = st.session_state['result']
name = res['name'] or "N/A"
usn = res['usn'] or "N/A"
prediction = res['prediction']
probs = res['probs']
vals = res['values']
confidence = probs[prediction]
conf_text = confidence_label(confidence)

# ---- Analysis, recommendations, model info, charts ----
status, analysis_rows = analyse(vals, avg)
recommendations = get_recommendations(prediction, vals, avg, status)

def get_algo_name(m):
    if hasattr(m, 'steps'):
        base_clf = m.steps[-1][1]
        name = type(base_clf).__name__
    else:
        name = type(m).__name__
    mapping = {
        'LogisticRegression': 'Logistic Regression (Standardized Pipeline)',
        'RandomForestClassifier': 'Random Forest Classifier',
        'GradientBoostingClassifier': 'Gradient Boosting Classifier',
        'DecisionTreeClassifier': 'Decision Tree Classifier'
    }
    return mapping.get(name, name)

model_info = {
    'algorithm': get_algo_name(model),
    'n_features': len(feature_cols),
    'n_classes': len(model.classes_),
    'dataset_size': len(df),
    'metrics': get_model_metrics(model, df, feature_cols),
}

fig_acad = make_academic_chart(vals, avg)
fig_eng = make_engagement_chart(vals, avg)

# ================= DASHBOARD DISPLAY =================
st.subheader("Student Profile Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Student Name", name)
c2.metric("USN / ID", usn)
c3.metric("GPA", f"{vals['previous_semester_gpa']:.2f}")
c4.metric("Attendance", f"{vals['attendance_percentage']:.1f}%")

st.markdown("---")

st.subheader("Prediction Result")
st.markdown(
    f"<div style='background-color:{CARD_BG[prediction]}; padding:20px; border-radius:12px; border-left: 6px solid {CARD_TEXT[prediction]};'>"
    f"<h2 style='color:{CARD_TEXT[prediction]}; margin:0;'>{ICONS[prediction]} {prediction.upper()} RISK</h2>"
    f"<p style='color:{CARD_TEXT[prediction]}; margin:8px 0 0 0; font-size:18px; font-weight: 500;'>"
    f"Model Confidence: {confidence:.1f}% ({conf_text} confidence)</p>"
    f"<p style='color:{CARD_TEXT[prediction]}; margin:4px 0 0 0;'>"
    f"The student exhibits indicators aligned with the {prediction} risk tier based on all 20 evaluated academic and engagement factors.</p></div>",
    unsafe_allow_html=True
)

st.write("")
st.subheader("Risk Probability Breakdown")
p1, p2 = st.columns([1, 1.5])
with p1:
    fig_donut = make_donut(probs, prediction, confidence)
    st.pyplot(fig_donut)
    plt.close(fig_donut)
with p2:
    st.write("#### Class Probabilities:")
    for c in CATEGORIES:
        prob_val = probs.get(c, 0)
        st.write(f"{ICONS[c]} **{c} Risk:** {prob_val}%")
        st.progress(min(max(prob_val / 100, 0.0), 1.0))

st.markdown("---")

st.subheader("🔍 Detailed Factor Analysis vs. Peer Cohort")
st.caption("Benchmark comparison evaluating the student's metrics across all 20 dimensions against dataset averages.")

df_analysis = pd.DataFrame([{
    "Factor": r['factor'],
    "Student Value": r['student'],
    "Dataset Average": r['average'],
    "Comparison": r['comparison'],
    "Status Indicator": STATUS_SCREEN[r['status']],
} for r in analysis_rows])

st.dataframe(df_analysis, hide_index=True, use_container_width=True)

st.markdown("---")

st.subheader("📊 Visual Factor Comparisons")
st.pyplot(fig_acad)
st.pyplot(fig_eng)

st.markdown("---")

st.subheader("💡 Tailored Action Plan & Recommendations")
for r in recommendations:
    st.markdown("- " + r)

st.markdown("---")

st.subheader("🤖 Machine Learning Model Performance & Diagnostics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Algorithm", model_info['algorithm'])
m2.metric("Evaluated Features", f"{model_info['n_features']} parameters")
m3.metric("Output Classes", f"{model_info['n_classes']} (Low/Med/High)")
m4.metric("Cohort Dataset Size", f"{model_info['dataset_size']} students")

metrics = model_info['metrics']
if metrics:
    st.write(f"**Cross-Validation Split:** {metrics['n_train']} training / {metrics['n_test']} test students")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test Accuracy", f"{metrics['Accuracy'] * 100:.2f}%")
    k2.metric("Weighted Precision", f"{metrics['Precision'] * 100:.2f}%")
    k3.metric("Weighted Recall", f"{metrics['Recall'] * 100:.2f}%")
    k4.metric("F1 Score", f"{metrics['F1 Score'] * 100:.2f}%")
    st.caption("Evaluated on stratified holdout test data demonstrating strong generalization across all risk tiers.")

st.markdown("---")

st.subheader("📄 Export Official Assessment Report")
pdf_file = build_pdf(
    name, usn, prediction, confidence, conf_text, probs, vals, status,
    res['extracurricular'], recommendations, model_info, [fig_acad, fig_eng]
)
st.download_button(
    "📥 Download Assessment Report (PDF)",
    data=pdf_file,
    file_name=f"risk_report_{usn if usn != 'N/A' else 'student'}.pdf",
    mime="application/pdf",
    type="primary"
)

plt.close(fig_acad)
plt.close(fig_eng)