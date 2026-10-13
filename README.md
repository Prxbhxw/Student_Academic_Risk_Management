# 🎓 Student Academic Risk Management & Prediction System

An end-to-end Machine Learning web application and early-warning decision support system designed to predict and prevent academic failure. The platform evaluates students across **20 multi-dimensional academic, behavioral, self-regulation, and practical engagement parameters**, achieving **97.92% accuracy** across all risk tiers (Low, Medium, High).

Live Dashboard: Built with [Streamlit](https://streamlit.io/)

---

## 🚀 Key Features

- **20-Parameter Risk Assessment:** Comprehensive modeling covering academic performance, previous course failure history, self-learning time, online study habits, tutoring, internships, and well-being.
- **Accurate Machine Learning Pipeline:** Standardized Pipeline (`StandardScaler` + `LogisticRegression`) trained with stratified cross-validation delivering **97.92% Test Accuracy** and **97.92% F1 Score**.
- **Interactive Decision Dashboard:**
  - Real-time risk tier prediction with confidence percentage.
  - Interactive risk class probability donut charts and progress distributions.
  - Granular factor-by-factor benchmark analysis comparing the student against the cohort average.
  - Dual-panel visual comparisons for academic performance and engagement.
  - Personalized, rule-based actionable recommendations.
  - **One-click Official PDF Report Export** with embedded comparison charts.
- **Production-Ready & Cloud-Optimized:** Seamless deployment on **Streamlit Community Cloud** with resilient dynamic path resolution and on-the-fly model fallback.

---

## 📊 Feature Parameters (20 Evaluated Dimensions)

### 1. Academic Performance & Course History
1. `attendance_percentage`: Attendance Percentage (0 – 100%)
2. `avg_test_score`: Average Test Score (0 – 100)
3. `assignments_submitted_pct`: Assignment Submission Percentage (0 – 100%)
4. `previous_semester_gpa`: Previous Semester GPA (0.0 – 10.0)
5. `backlogs`: Number of Active Backlogs (0 – 5)
6. `prev_failed_courses`: Previous Semester Failed Courses (0 – 5)

### 2. Study Habits & Self-Regulation
7. `study_hours_per_week`: Weekly Study Hours (0 – 40 hrs/wk)
8. `weekly_self_learning_hours`: Weekly Self-Learning Time (0 – 35 hrs/wk)
9. `online_learning_hours`: Online Learning Duration (0 – 30 hrs/wk)
10. `library_hours_per_week`: Library Study Hours per Week (0 – 20 hrs/wk)
11. `learning_plan_completion_pct`: Personal Learning Plan Completion (0 – 100%)

### 3. Tutoring, Engagement & Well-Being
12. `tutoring_participation`: Academic Tutoring Participation (No / Yes)
13. `discussion_participation_freq`: Academic Discussion Participation Frequency (0 – 10 sessions/wk)
14. `exercise_hours_per_week`: Exercise Duration and Frequency (0 – 14 hrs/wk)
15. `extracurricular_participation`: Extracurricular Activity Participation (No / Yes)
16. `num_activities`: Number of Activities / Clubs (0 – 5)
17. `achievements`: Achievements / Prizes Won (0 – 5)

### 4. Practical & Competition Experience
18. `competition_participation`: Subject-Related Competition Participation (0 – 5)
19. `internship_experience`: Internship and Practical Experience (No / Yes)
20. `practical_opportunity_freq`: Internship & Practical Opportunity Frequency (0 – 5)

---

## 🤖 Model Performance

Evaluated on a 20% stratified holdout test set (240 students) from a cohort dataset of 1,200 student records:

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **97.92%** |
| **Precision (Weighted)** | **97.93%** |
| **Recall (Weighted)** | **97.92%** |
| **F1 Score (Weighted)** | **97.92%** |

### Confusion Matrix
- **High Risk:** 42 / 43 correct (97.7% recall)
- **Medium Risk:** 87 / 89 correct (97.8% recall)
- **Low Risk:** 106 / 108 correct (98.1% recall)

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── config.toml          # Streamlit headless and port configuration
├── app/
│   ├── analysis.py          # Factor status benchmarking and recommendations
│   ├── charts.py            # Comparison visualizations and donut chart
│   ├── dashboard.py         # Main interactive Streamlit application
│   ├── report.py            # PDF report generator with matplotlib
│   └── risk_model.pkl       # Serialized trained machine learning model
├── data/
│   └── students_dataset.csv # Cohort dataset (1,200 records, 20 features)
├── notebooks/
│   └── exploration.ipynb    # Data exploration and model comparison notebook
├── src/
│   ├── generate_data.py     # Synthetic dataset generator
│   └── train_model.py       # Model training, benchmarking, and export script
├── .gitignore
├── requirements.txt         # Python package dependencies
├── streamlit_app.py         # Root entry point for Streamlit Community Cloud
└── README.md
```

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Prxbhxw/Student_Academic_Risk_Management.git
   cd Student_Academic_Risk_Management
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run streamlit_app.py
   # or
   streamlit run app/dashboard.py
   ```

4. Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Fork or push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** and select:
   - **Repository:** `Prxbhxw/Student_Academic_Risk_Management`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Click **Deploy!**