import os
import sys
import importlib
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
    import analysis
    importlib.reload(analysis)
    from analysis import analyse, confidence_label, get_recommendations, get_model_metrics
    import charts
    importlib.reload(charts)
    from charts import CATEGORIES, make_donut, make_academic_chart, make_engagement_chart
    import report
    importlib.reload(report)
    from report import build_pdf
    import auth
    importlib.reload(auth)
    from auth import authenticate
except ImportError:
    from app import analysis, charts, report, auth
    importlib.reload(analysis)
    importlib.reload(charts)
    importlib.reload(report)
    importlib.reload(auth)
    from app.analysis import analyse, confidence_label, get_recommendations, get_model_metrics
    from app.charts import CATEGORIES, make_donut, make_academic_chart, make_engagement_chart
    from app.report import build_pdf
    from app.auth import authenticate

st.set_page_config(page_title="Student Academic Risk Predictor & Portal", layout="wide", page_icon="🎓")

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
    st.error("Error: Could not locate 'students_dataset.csv'. Please ensure 'data/students_dataset.csv' exists.")
    st.stop()

df = pd.read_csv(DATA_PATH)
avg = df[feature_cols].mean()

# ---- Load model ----
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

# ----------------- SESSION STATE & AUTH INITIALIZATION -----------------
if 'user' not in st.session_state:
    st.session_state['user'] = None

# Helper to compute prediction dictionary for a student row/dict
def compute_prediction_for_values(values, name, usn):
    input_data = pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    return {
        'name': name,
        'usn': usn,
        'prediction': prediction,
        'probs': {cls: round(float(p) * 100, 1) for cls, p in zip(model.classes_, proba)},
        'values': values,
        'extracurricular': f"{values.get('extracurricular_participation', 0):g} hrs/wk",
    }

# ----------------- LOGIN / PORTAL SELECTION PAGE -----------------
def render_login_page():
    st.markdown(
        """
        <style>
        .login-card {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            margin-top: 10px;
        }
        .login-header {
            text-align: center;
            margin-bottom: 25px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div class='login-header'>
                <h1 style='font-size: 2.2rem; margin-bottom: 5px;'>🎓 Academic Risk Portal</h1>
                <p style='color: #6c757d; font-size: 1.05rem;'>Unified Academic Early Warning & Quantitative Risk Prediction System</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        role_tab = st.radio(
            "Select Access Mode",
            [
                "🌐 Guest / Free Self-Assessment (No Login Required)",
                "👨‍🏫 Teacher / College Admin",
                "👨‍🎓 Registered Student"
            ],
            index=0,
            horizontal=False
        )

        if "Guest" in role_tab:
            st.markdown(
                """
                <div style='background-color:#e8f4fd; padding:15px; border-radius:10px; border-left: 5px solid #007bff; margin-bottom: 18px;'>
                    <h4 style='color:#0056b3; margin:0 0 5px 0;'>🌐 Free Individual Self-Assessment</h4>
                    <p style='color:#333; margin:0; font-size: 14px;'>
                        No login or Student ID required. Enter your details directly, input all quantitative factors, 
                        and generate a comprehensive diagnostic risk report immediately.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            guest_name = st.text_input("Enter Your Name (Optional)", value="", placeholder="e.g. Alex Morgan")
            if st.button("🚀 Start Free Self-Assessment", type="primary", use_container_width=True):
                display_name = guest_name.strip() if guest_name.strip() else "Guest User"
                st.session_state['user'] = {
                    "username": "guest",
                    "role": "guest",
                    "name": display_name,
                    "department": "Public Learner",
                    "college": "Self-Assessment Portal",
                    "student_id": "GUEST"
                }
                st.session_state.pop('result', None)
                st.rerun()

        else:
            selected_role = "teacher" if "Teacher" in role_tab else "student"
            with st.form("login_form"):
                if selected_role == "teacher":
                    st.subheader("🏛️ Faculty & Admin Sign In")
                    username = st.text_input("Faculty ID / Username", placeholder="e.g. admin or teacher")
                    password = st.text_input("Password", type="password", placeholder="Enter password")
                else:
                    st.subheader("🎓 Student Sign In")
                    username = st.text_input("Student USN / Roll No.", placeholder="e.g. S1000, S1002, S1013")
                    password = st.text_input("Password", type="password", placeholder="Enter password")

                submit = st.form_submit_button("Sign In 🔐", use_container_width=True, type="primary")

                if submit:
                    user_record, error_msg = authenticate(username, password, selected_role)
                    if error_msg:
                        st.error(f"❌ {error_msg}")
                    else:
                        st.session_state['user'] = user_record
                        st.session_state['selected_student_id'] = user_record.get('student_id')
                        st.session_state.pop('result', None)
                        st.success(f"Welcome, {user_record['name']}!")
                        st.rerun()

            st.markdown("---")
            with st.expander("🔑 Quick Demo Credentials (Click to view)"):
                st.markdown(
                    """
                    **👨‍🏫 Teacher / Admin Portal:**
                    - Username: `admin` | Password: `admin123` *(Dr. Sarah Mitchell - Dean)*
                    - Username: `teacher` | Password: `admin123` *(Prof. Alan Turing - CS Dept)*

                    **👨‍🎓 Student Portal:**
                    - Student 1: `S1000` | Password: `student123` *(Alex Morgan - Medium Risk)*
                    - Student 2: `S1002` | Password: `student123` *(Emily Watson - Low Risk)*
                    - Student 3: `S1013` | Password: `student123` *(Daniel Craig - High Risk)*
                    - *(Any Student ID from `S1000` to `S2199` works with `student123`)*

                    **🌐 Normal User / Guest:**
                    - Select **"Guest / Free Self-Assessment (No Login Required)"** to test without any credentials!
                    """
                )

# Check login state
if st.session_state['user'] is None:
    render_login_page()
    st.stop()

# ----------------- LOGGED IN APPLICATION -----------------
current_user = st.session_state['user']
is_admin = current_user['role'] in ['teacher', 'admin']
is_guest = current_user['role'] == 'guest'

# ----------------- SIDEBAR HEADER & LOGOUT -----------------
with st.sidebar:
    if is_admin:
        role_label = "👨‍🏫 Admin & Faculty Control"
    elif is_guest:
        role_label = "🌐 Guest Self-Assessment"
    else:
        role_label = "👨‍🎓 Student Portal"

    st.markdown(f"### {role_label}")
    st.markdown(f"**Current User:** {current_user['name']}")
    st.caption(f"{current_user['department']} | {current_user['college']}")
    
    if st.button("🚪 Exit / Switch Role", use_container_width=True):
        st.session_state['user'] = None
        st.session_state.pop('result', None)
        st.session_state.pop('selected_student_id', None)
        st.rerun()

    st.markdown("---")

# ----------------- ROLE FLOWS -----------------
if is_admin:
    # ---------------- TEACHER / ADMIN FLOW ----------------
    with st.sidebar:
        st.header("🎯 Cohort Controls")
        
        mode = st.radio(
            "Select View",
            ["📊 Class Roster & Student Picker", "🎛️ Manual Simulator / Prediction"],
            index=0
        )
        st.markdown("---")
        
    if mode == "📊 Class Roster & Student Picker":
        st.title("🏛️ Faculty Class Roster & Student Profile Picker")
        st.write("Browse your class or department, filter by predicted risk category, and select any student to drill down into their full diagnostic assessment.")
        
        @st.cache_data
        def get_cohort_summary():
            X = df[feature_cols]
            preds = model.predict(X)
            summary_df = df[['student_id', 'attendance_percentage', 'avg_test_score', 'previous_semester_gpa', 'backlogs', 'risk_label']].copy()
            summary_df['predicted_risk'] = preds
            return summary_df

        cohort_df = get_cohort_summary()

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        total_students = len(cohort_df)
        high_risk_count = (cohort_df['predicted_risk'] == 'High').sum()
        med_risk_count = (cohort_df['predicted_risk'] == 'Medium').sum()
        low_risk_count = (cohort_df['predicted_risk'] == 'Low').sum()

        col_m1.metric("Total Class Size", f"{total_students} Students")
        col_m2.metric("🔴 High Risk Alert", f"{high_risk_count} ({high_risk_count/total_students*100:.1f}%)")
        col_m3.metric("🟡 Medium Risk", f"{med_risk_count} ({med_risk_count/total_students*100:.1f}%)")
        col_m4.metric("🟢 Low Risk", f"{low_risk_count} ({low_risk_count/total_students*100:.1f}%)")

        st.markdown("---")

        st.subheader("🔍 Class Filter & Student Selection")
        f1, f2, f3 = st.columns([1, 1, 2])
        with f1:
            risk_filter = st.selectbox("Filter by Risk Tier", ["All Tiers", "High", "Medium", "Low"])
        with f2:
            search_query = st.text_input("Search Student ID", placeholder="e.g. S1013").strip().upper()
        
        filtered_df = cohort_df
        if risk_filter != "All Tiers":
            filtered_df = filtered_df[filtered_df['predicted_risk'] == risk_filter]
        if search_query:
            filtered_df = filtered_df[filtered_df['student_id'].str.contains(search_query)]

        with f3:
            options_list = filtered_df['student_id'].tolist()
            if options_list:
                selected_id = st.selectbox("Select Student Profile to Inspect:", options_list)
            else:
                st.warning("No students match your filter criteria.")
                selected_id = None

        with st.expander(f"📋 View Filtered Class Table ({len(filtered_df)} students)", expanded=False):
            st.dataframe(
                filtered_df.rename(columns={
                    'student_id': 'USN / ID',
                    'attendance_percentage': 'Attendance %',
                    'avg_test_score': 'Test Score',
                    'previous_semester_gpa': 'GPA',
                    'backlogs': 'Active Backlogs',
                    'predicted_risk': 'Predicted Risk'
                }),
                hide_index=True,
                use_container_width=True
            )

        if selected_id:
            row = df[df['student_id'] == selected_id].iloc[0]
            student_values = {col: float(row[col]) for col in feature_cols}
            st.session_state['result'] = compute_prediction_for_values(
                student_values,
                name=f"Student {selected_id}",
                usn=selected_id
            )

    else:
        st.title("🎛️ Manual Risk Assessment & Simulator")
        st.write("Simulate or manually adjust any student's 20 parameters to evaluate hypothetical risk and intervention scenarios.")

        with st.sidebar:
            st.subheader("📋 Student Inputs")
            with st.expander("👤 Student Information", expanded=True):
                sim_name = st.text_input("Student Name", placeholder="e.g. Alex Morgan", value="Alex Morgan")
                sim_usn = st.text_input("USN / Student ID", placeholder="e.g. 1MS21CS001", value="1MS21CS001")

            with st.expander("📚 Primary Academic Pillars (High Impact)", expanded=True):
                test_score = st.slider("Average Test Score (0-100) [18% Power]", 0, 100, 65)
                gpa = st.slider("Previous Semester GPA (0.0 - 10.0) [19% Power]", 0.0, 10.0, 6.5, step=0.1)
                attendance = st.slider("Attendance Percentage (%) [18% Power]", 0, 100, 75)
                prev_failed = st.slider("Previous Failed Courses [Severe -15% Power]", 0, 5, 0)
                backlogs = st.slider("Number of Active Backlogs [Severe -13% Power]", 0, 5, 0)
                assignments = st.slider("Assignments Submitted (%) [9% Power]", 0, 100, 80)
                learning_plan_pct = st.slider("Personal Learning Plan Completion (%) [8% Power]", 0, 100, 75)

            with st.expander("⏱️ Study Habits & Self-Regulation", expanded=False):
                study_hours = st.slider("Study Hours per Week (hrs/wk) [5% Power]", 0, 40, 12)
                weekly_self_learning = st.slider("Weekly Self-Learning Time (hrs/wk) [4% Power]", 0, 35, 11)
                online_learning = st.slider("Online Learning Duration (hrs/wk) [3% Power]", 0, 30, 8)
                library_hours = st.slider("Library Hours per Week (hrs/wk) [3% Power]", 0, 20, 6)

            with st.expander("💼 Mentorship & Practical Experience (Quantified)", expanded=False):
                internship_duration = st.slider("Internship Experience Duration (Months) [5% Power]", 0, 12, 1)
                tutoring_sessions = st.slider("Tutoring Sessions Attended (Sessions/Month) [5% Power]", 0, 12, 2)
                practical_freq = st.slider("Practical / Hands-on Opportunity Frequency [4% Power]", 0, 5, 1)
                competition_part = st.slider("Subject Competitions & Hackathons Count [4% Power]", 0, 5, 1)

            with st.expander("🤝 Engagement, Clubs & Wellness (Quantified)", expanded=False):
                extracurricular_hours = st.slider("Extracurricular Engagement Time (hrs/wk) [3% Power]", 0.0, 15.0, 3.0, step=0.5)
                discussion_freq = st.slider("Academic Discussion Participation Frequency (/wk) [3% Power]", 0.0, 10.0, 4.0, step=0.5)
                achievements = st.slider("Achievements & Awards Won [2% Power]", 0, 5, 1)
                num_activities = st.slider("Active Clubs / Societies Count [1% Power]", 0, 5, 1)
                exercise_hours = st.slider("Exercise & Physical Activity (hrs/wk) [1% Power]", 0.0, 14.0, 5.0, step=0.5)

            predict_clicked = st.button("🚀 Run Simulator Assessment", type="primary", use_container_width=True)

            if predict_clicked:
                sim_values = {
                    'attendance_percentage': attendance,
                    'avg_test_score': test_score,
                    'assignments_submitted_pct': assignments,
                    'study_hours_per_week': study_hours,
                    'previous_semester_gpa': gpa,
                    'extracurricular_participation': extracurricular_hours,
                    'backlogs': backlogs,
                    'library_hours_per_week': library_hours,
                    'achievements': achievements,
                    'num_activities': num_activities,
                    'prev_failed_courses': prev_failed,
                    'online_learning_hours': online_learning,
                    'tutoring_participation': tutoring_sessions,
                    'competition_participation': competition_part,
                    'internship_experience': internship_duration,
                    'learning_plan_completion_pct': learning_plan_pct,
                    'weekly_self_learning_hours': weekly_self_learning,
                    'practical_opportunity_freq': practical_freq,
                    'discussion_participation_freq': discussion_freq,
                    'exercise_hours_per_week': exercise_hours,
                }
                st.session_state['result'] = compute_prediction_for_values(sim_values, sim_name, sim_usn)

elif is_guest:
    # ---------------- GUEST / NORMAL USER FLOW ----------------
    st.title("🌐 Individual Academic Self-Assessment & Risk Report")
    st.write(
        f"Welcome, **{current_user['name']}**! Enter your quantitative academic metrics below. "
        "Every single parameter holds a calibrated decision-making weight in calculating your risk outcome."
    )

    with st.form("guest_assessment_form"):
        st.subheader("📝 Input Your Academic Metrics")
        
        c1, c2 = st.columns(2)
        with c1:
            eval_name = st.text_input("Name", value=current_user['name'], placeholder="Your Full Name")
        with c2:
            eval_id = st.text_input("Identifier / Roll No (Optional)", value="SELF-EVAL", placeholder="e.g. 2026-CS-01")

        st.markdown("#### 1. Core Academic Pillars (Highest Decision Power ~60%)")
        st.caption("These primary metrics hold the strongest weight in determining your academic standing.")
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            test_score = st.slider("Average Test Score (0 - 100)", 0, 100, 72, help="Decision Power: 18%")
            attendance = st.slider("Attendance Percentage (%)", 0, 100, 80, help="Decision Power: 18%")
        with col_a2:
            gpa = st.slider("Current / Previous GPA (0.0 - 10.0)", 0.0, 10.0, 7.0, step=0.1, help="Decision Power: 19%")
            prev_failed = st.slider("Previous Semester Failed Courses", 0, 5, 0, help="Severe Penalty: -15%")
        with col_a3:
            assignments = st.slider("Assignments Submitted (%)", 0, 100, 85, help="Decision Power: 9%")
            backlogs = st.slider("Active Backlogs Count", 0, 5, 0, help="Severe Penalty: -13%")

        st.markdown("#### 2. Rigor & Study Habits (Power ~20%)")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            learning_plan_pct = st.slider("Personal Learning Plan Done (%)", 0, 100, 80, help="Decision Power: 8%")
            study_hours = st.slider("Study Hours / Week", 0, 40, 15, help="Decision Power: 5%")
        with col_s2:
            weekly_self_learning = st.slider("Weekly Self-Learning (hrs/wk)", 0, 35, 12, help="Decision Power: 4%")
            library_hours = st.slider("Library Hours / Week", 0, 20, 6, help="Decision Power: 3%")
        with col_s3:
            online_learning = st.slider("Online Digital Study (hrs/wk)", 0, 30, 8, help="Decision Power: 3%")

        st.markdown("#### 3. Quantified Mentorship, Practical Experience & Engagement (Power ~20%)")
        st.caption("Each month of internship, tutoring session, and club hour quantifiably reduces academic risk.")
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            internship_duration = st.slider("Internship Duration (Months Completed)", 0, 12, 2, help="Decision Power: 5%")
            tutoring_sessions = st.slider("Tutoring Attended (Sessions/Month)", 0, 12, 3, help="Decision Power: 5%")
        with col_e2:
            competition_part = st.slider("Competitions & Hackathons Count", 0, 5, 1, help="Decision Power: 4%")
            practical_freq = st.slider("Practical Opportunities Frequency", 0, 5, 2, help="Decision Power: 4%")
        with col_e3:
            extracurricular_hours = st.slider("Extracurricular Activity (hrs/wk)", 0.0, 15.0, 3.5, step=0.5, help="Decision Power: 3%")
            discussion_freq = st.slider("Discussion Frequency (/wk)", 0.0, 10.0, 4.0, step=0.5, help="Decision Power: 3%")

        col_x1, col_x2, col_x3 = st.columns(3)
        with col_x1:
            achievements = st.slider("Achievements / Prizes Won", 0, 5, 1, help="Decision Power: 2%")
        with col_x2:
            num_activities = st.slider("Clubs / Technical Societies Count", 0, 5, 2, help="Decision Power: 1%")
        with col_x3:
            exercise_hours = st.slider("Fitness & Exercise (hrs/wk)", 0.0, 14.0, 5.0, step=0.5, help="Decision Power: 1%")

        submit_guest = st.form_submit_button("🚀 Calculate Risk & Generate Diagnostic Report", type="primary", use_container_width=True)

    if submit_guest or ('result' not in st.session_state or st.session_state['result'] is None):
        guest_values = {
            'attendance_percentage': attendance,
            'avg_test_score': test_score,
            'assignments_submitted_pct': assignments,
            'study_hours_per_week': study_hours,
            'previous_semester_gpa': gpa,
            'extracurricular_participation': extracurricular_hours,
            'backlogs': backlogs,
            'library_hours_per_week': library_hours,
            'achievements': achievements,
            'num_activities': num_activities,
            'prev_failed_courses': prev_failed,
            'online_learning_hours': online_learning,
            'tutoring_participation': tutoring_sessions,
            'competition_participation': competition_part,
            'internship_experience': internship_duration,
            'learning_plan_completion_pct': learning_plan_pct,
            'weekly_self_learning_hours': weekly_self_learning,
            'practical_opportunity_freq': practical_freq,
            'discussion_participation_freq': discussion_freq,
            'exercise_hours_per_week': exercise_hours,
        }
        st.session_state['result'] = compute_prediction_for_values(guest_values, eval_name, eval_id)

else:
    # ---------------- STUDENT FLOW ----------------
    student_id = current_user.get('student_id')
    st.title("🎓 My Academic Risk & Performance Portal")
    st.write(f"Welcome, **{current_user['name']}** ({student_id})! Track your current semester performance or update your metrics to test new results.")

    matching = df[df['student_id'] == student_id]
    if not matching.empty:
        row = matching.iloc[0]
        base_vals = {col: float(row[col]) for col in feature_cols}
    else:
        base_vals = {col: float(avg[col]) for col in feature_cols}

    # Choice to view current profile or update/simulate new metrics
    student_view_mode = st.radio(
        "Select Performance View:",
        ["📌 View Official Current Semester Record", "✏️ Update / Simulate New Academic Performance"],
        horizontal=True
    )

    if student_view_mode == "📌 View Official Current Semester Record":
        st.caption("Displaying your registered academic profile stored in the college database.")
        st.session_state['result'] = compute_prediction_for_values(
            base_vals,
            name=current_user['name'],
            usn=student_id
        )
    else:
        st.info("💡 **Interactive Update Mode**: Adjust any of your scores, study hours, or activities below to evaluate how your new academic efforts impact your predicted risk tier.")
        with st.form("student_update_form"):
            st.subheader("📝 Update Your Current Performance Metrics")

            st.markdown("#### 1. Core Academic Pillars (Highest Decision Power ~60%)")
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                u_test_score = st.slider("Average Test Score (0 - 100)", 0, 100, int(round(base_vals.get('avg_test_score', 70))), help="18% Power")
                u_attendance = st.slider("Attendance Percentage (%)", 0, 100, int(round(base_vals.get('attendance_percentage', 75))), help="18% Power")
            with col_a2:
                u_gpa = st.slider("Current / Previous GPA (0.0 - 10.0)", 0.0, 10.0, float(round(base_vals.get('previous_semester_gpa', 6.5), 1)), step=0.1, help="19% Power")
                u_prev_failed = st.slider("Previous Failed Courses", 0, 5, int(base_vals.get('prev_failed_courses', 0)), help="Penalty: -15%")
            with col_a3:
                u_assignments = st.slider("Assignments Submitted (%)", 0, 100, int(round(base_vals.get('assignments_submitted_pct', 80))), help="9% Power")
                u_backlogs = st.slider("Active Backlogs Count", 0, 5, int(base_vals.get('backlogs', 0)), help="Penalty: -13%")

            st.markdown("#### 2. Rigor & Study Habits (Power ~20%)")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                u_learning_plan = st.slider("Personal Learning Plan Done (%)", 0, 100, int(round(base_vals.get('learning_plan_completion_pct', 75))), help="8% Power")
                u_study_hours = st.slider("Study Hours / Week", 0, 40, int(round(base_vals.get('study_hours_per_week', 12))), help="5% Power")
            with col_s2:
                u_self_learn = st.slider("Weekly Self-Learning (hrs/wk)", 0, 35, int(round(base_vals.get('weekly_self_learning_hours', 10))), help="4% Power")
                u_library = st.slider("Library Hours / Week", 0, 20, int(round(base_vals.get('library_hours_per_week', 5))), help="3% Power")
            with col_s3:
                u_online = st.slider("Online Digital Study (hrs/wk)", 0, 30, int(round(base_vals.get('online_learning_hours', 8))), help="3% Power")

            st.markdown("#### 3. Mentorship, Practical Experience & Engagement (Power ~20%)")
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                u_internship = st.slider("Internship Duration (Months Completed)", 0, 12, int(base_vals.get('internship_experience', 1)), help="5% Power")
                u_tutoring = st.slider("Tutoring Attended (Sessions/Month)", 0, 12, int(base_vals.get('tutoring_participation', 2)), help="5% Power")
            with col_e2:
                u_competition = st.slider("Competitions & Hackathons Count", 0, 5, int(base_vals.get('competition_participation', 1)), help="4% Power")
                u_practical = st.slider("Practical Opportunities Frequency", 0, 5, int(base_vals.get('practical_opportunity_freq', 1)), help="4% Power")
            with col_e3:
                u_extracurricular = st.slider("Extracurricular Activity (hrs/wk)", 0.0, 15.0, float(round(base_vals.get('extracurricular_participation', 3.0), 1)), step=0.5, help="3% Power")
                u_discussion = st.slider("Discussion Frequency (/wk)", 0.0, 10.0, float(round(base_vals.get('discussion_participation_freq', 4.0), 1)), step=0.5, help="3% Power")

            col_x1, col_x2, col_x3 = st.columns(3)
            with col_x1:
                u_achievements = st.slider("Achievements / Prizes Won", 0, 5, int(base_vals.get('achievements', 0)), help="2% Power")
            with col_x2:
                u_activities = st.slider("Clubs / Technical Societies Count", 0, 5, int(base_vals.get('num_activities', 1)), help="1% Power")
            with col_x3:
                u_exercise = st.slider("Fitness & Exercise (hrs/wk)", 0.0, 14.0, float(round(base_vals.get('exercise_hours_per_week', 5.0), 1)), step=0.5, help="1% Power")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                update_submit = st.form_submit_button("🚀 Preview Updated Risk Report", use_container_width=True)
            with col_btn2:
                save_to_db = st.form_submit_button("💾 Save Permanently to Official Record", type="primary", use_container_width=True)

        if update_submit or save_to_db or ('result' not in st.session_state or st.session_state['result'] is None):
            updated_vals = {
                'attendance_percentage': float(u_attendance),
                'avg_test_score': float(u_test_score),
                'assignments_submitted_pct': float(u_assignments),
                'study_hours_per_week': float(u_study_hours),
                'previous_semester_gpa': float(u_gpa),
                'extracurricular_participation': float(u_extracurricular),
                'backlogs': int(u_backlogs),
                'library_hours_per_week': float(u_library),
                'achievements': int(u_achievements),
                'num_activities': int(u_activities),
                'prev_failed_courses': int(u_prev_failed),
                'online_learning_hours': float(u_online),
                'tutoring_participation': int(u_tutoring),
                'competition_participation': int(u_competition),
                'internship_experience': int(u_internship),
                'learning_plan_completion_pct': float(u_learning_plan),
                'weekly_self_learning_hours': float(u_self_learn),
                'practical_opportunity_freq': int(u_practical),
                'discussion_participation_freq': float(u_discussion),
                'exercise_hours_per_week': float(u_exercise),
            }
            computed = compute_prediction_for_values(
                updated_vals,
                name=current_user['name'],
                usn=student_id
            )
            st.session_state['result'] = computed

            if save_to_db:
                # Update persistent dataset file
                mask = df['student_id'] == student_id
                if mask.any():
                    for col_k, val_v in updated_vals.items():
                        df.loc[mask, col_k] = val_v
                    df.loc[mask, 'risk_label'] = computed['prediction']
                else:
                    new_row = {'student_id': student_id, **updated_vals, 'risk_label': computed['prediction']}
                    df.loc[len(df)] = new_row
                
                df.to_csv(DATA_PATH, index=False)
                st.cache_data.clear()
                st.success("✅ Changes successfully saved to the official database! Your profile and faculty roster are now permanently updated.")

# ----------------- DISPLAY RESULTS IF AVAILABLE -----------------
if 'result' in st.session_state and st.session_state['result'] is not None:
    res = st.session_state['result']
    name = res['name'] or "N/A"
    usn = res['usn'] or "N/A"
    prediction = res['prediction']
    probs = res['probs']
    vals = res['values']
    confidence = probs[prediction]
    conf_text = confidence_label(confidence)

    status, analysis_rows = analyse(vals, avg)
    recommendations = get_recommendations(prediction, vals, avg, status)

    fig_acad = make_academic_chart(vals, avg)
    fig_eng = make_engagement_chart(vals, avg)

    st.markdown("---")
    st.subheader(f"📌 Diagnostic Report: {name} ({usn})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluated Name", name)
    c2.metric("Identifier / ID", usn)
    c3.metric("GPA (19% Power)", f"{vals['previous_semester_gpa']:.2f}")
    c4.metric("Attendance (18% Power)", f"{vals['attendance_percentage']:.1f}%")

    st.markdown("---")

    st.subheader("Prediction Result & Confidence")
    st.markdown(
        f"<div style='background-color:{CARD_BG[prediction]}; padding:20px; border-radius:12px; border-left: 6px solid {CARD_TEXT[prediction]};'>"
        f"<h2 style='color:{CARD_TEXT[prediction]}; margin:0;'>{ICONS[prediction]} {prediction.upper()} RISK</h2>"
        f"<p style='color:{CARD_TEXT[prediction]}; margin:8px 0 0 0; font-size:18px; font-weight: 500;'>"
        f"Model Confidence: {confidence:.1f}% ({conf_text} confidence)</p>"
        f"<p style='color:{CARD_TEXT[prediction]}; margin:4px 0 0 0;'>"
        f"The evaluation exhibits indicators aligned with the {prediction} risk tier based on all 20 evaluated quantitative factors and decision weights.</p></div>",
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

    st.subheader("🔍 Factor Benchmark Analysis & Decision Impact Tiers")
    st.caption("Benchmark comparison evaluating student values against cohort averages, along with each parameter's decision-making power.")

    df_analysis = pd.DataFrame([{
        "Factor": r['factor'],
        "Student Value": r['student'],
        "Cohort Average": r['average'],
        "Comparison": r['comparison'],
        "Decision Impact Tier": r.get('power', 'Standard Parameter'),
        "Status Indicator": STATUS_SCREEN[r['status']],
    } for r in analysis_rows])

    st.dataframe(df_analysis, hide_index=True, use_container_width=True)

    st.markdown("---")

    st.subheader("📊 Visual Factor Comparisons")
    st.pyplot(fig_acad)
    st.pyplot(fig_eng)

    st.markdown("---")

    st.subheader("💡 Prioritized Action Plan & Targeted Interventions")
    for r in recommendations:
        st.markdown("- " + r)

    st.markdown("---")

    st.subheader("🤖 Machine Learning Model Diagnostics & Feature Power")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Algorithm", model_info['algorithm'])
    m2.metric("Evaluated Features", f"{model_info['n_features']} parameters (100% Quantitative)")
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
else:
    if is_admin:
        st.info("👈 Please select a student profile from the class picker or use the simulator to view results.")
