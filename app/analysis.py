from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# (column name, label shown on screen, type of factor)
ANALYSIS_ITEMS = [
    ('attendance_percentage', 'Attendance (%)', 'main'),
    ('avg_test_score', 'Test Score', 'main'),
    ('assignments_submitted_pct', 'Assignment Submission (%)', 'main'),
    ('previous_semester_gpa', 'Previous Semester GPA', 'main'),
    ('backlogs', 'Active Backlogs', 'backlog'),
    ('prev_failed_courses', 'Failed Courses (Prev Sem)', 'backlog'),
    ('study_hours_per_week', 'Study Hours / Week', 'main'),
    ('weekly_self_learning_hours', 'Self-Learning Hours / Week', 'main'),
    ('learning_plan_completion_pct', 'Learning Plan Completion (%)', 'main'),
    ('online_learning_hours', 'Online Learning Hours / Week', 'main'),
    ('library_hours_per_week', 'Library Hours / Week', 'main'),
    ('tutoring_participation', 'Academic Tutoring', 'engage'),
    ('competition_participation', 'Competitions Participated', 'engage'),
    ('internship_experience', 'Internship Experience', 'engage'),
    ('practical_opportunity_freq', 'Practical Opportunities', 'engage'),
    ('discussion_participation_freq', 'Discussion Frequency / Week', 'main'),
    ('exercise_hours_per_week', 'Exercise Hours / Week', 'main'),
    ('extracurricular_participation', 'Extracurricular Participation', 'engage'),
    ('num_activities', 'Activities / Clubs', 'engage'),
    ('achievements', 'Achievements / Prizes', 'engage'),
]


def get_status(kind, s, a):
    """Returns 'pos', 'neu' or 'bad' by comparing the student's value with the average."""
    if kind == 'backlog':
        if s == 0:
            return 'pos'
        return 'neu' if s <= a else 'bad'
    if kind == 'engage':
        # extra activities/prizes/tutoring/competitions/internships are a bonus
        return 'pos' if s >= a else 'neu'
    ratio = s / a if a > 0 else 1
    if ratio >= 1.05:
        return 'pos'
    if ratio >= 0.90:
        return 'neu'
    return 'bad'


def get_comparison(s, a):
    if a == 0:
        return "Above average" if s > 0 else "Around average"
    if s > a * 1.05:
        return "Above average"
    if s < a * 0.95:
        return "Below average"
    return "Around average"


def analyse(vals, avg):
    """Returns (status dictionary, list of rows for the analysis table)."""
    status = {}
    rows = []
    for col, label, kind in ANALYSIS_ITEMS:
        s, a = vals[col], avg[col]
        status[col] = get_status(kind, s, a)
        rows.append({'factor': label, 'student': f"{s:g}", 'average': f"{a:.1f}",
                     'comparison': get_comparison(s, a), 'status': status[col]})
    return status, rows


def confidence_label(p):
    if p >= 70:
        return "high"
    if p >= 50:
        return "moderate"
    return "low (borderline case)"


def get_recommendations(pred, vals, avg, status):
    """Recommendations are built from the student's real values across all parameters."""
    recs = []
    if status['attendance_percentage'] == 'bad':
        recs.append(f"Improve attendance: it is {vals['attendance_percentage']:g}% compared with the dataset average of {avg['attendance_percentage']:.1f}%.")
    if status['avg_test_score'] == 'bad':
        recs.append(f"Focus on improving test scores (yours: {vals['avg_test_score']:g}, dataset average: {avg['avg_test_score']:.1f}) and prepare strategically for assessments.")
    if status['assignments_submitted_pct'] == 'bad':
        recs.append(f"Submit assignments consistently (yours: {vals['assignments_submitted_pct']:g}%, dataset average: {avg['assignments_submitted_pct']:.1f}%).")
    if status['previous_semester_gpa'] == 'bad':
        recs.append(f"Work on boosting GPA (yours: {vals['previous_semester_gpa']:g}, dataset average: {avg['previous_semester_gpa']:.1f}).")
    if status['backlogs'] == 'bad' or (pred == "High" and vals['backlogs'] > 0):
        recs.append(f"Clear pending backlogs (currently {vals['backlogs']:g}) through remedial coursework.")
    if status['prev_failed_courses'] == 'bad' or (pred in ["High", "Medium"] and vals['prev_failed_courses'] > 0):
        recs.append(f"Enroll in recovery modules to clear previous semester failed course(s) (currently {vals['prev_failed_courses']:g}).")
    if status['study_hours_per_week'] == 'bad':
        recs.append(f"Increase weekly study hours (yours: {vals['study_hours_per_week']:g} hrs, dataset average: {avg['study_hours_per_week']:.1f} hrs).")
    if status['weekly_self_learning_hours'] == 'bad':
        recs.append(f"Dedicate more independent self-learning time (yours: {vals['weekly_self_learning_hours']:g} hrs/week, dataset average: {avg['weekly_self_learning_hours']:.1f} hrs/week).")
    if status['learning_plan_completion_pct'] == 'bad':
        recs.append(f"Improve personal learning plan completion (yours: {vals['learning_plan_completion_pct']:g}%, dataset average: {avg['learning_plan_completion_pct']:.1f}%).")
    if status['online_learning_hours'] == 'bad':
        recs.append(f"Leverage digital online learning resources (yours: {vals['online_learning_hours']:g} hrs/week, dataset average: {avg['online_learning_hours']:.1f} hrs/week).")
    if status['library_hours_per_week'] == 'bad':
        recs.append(f"Spend more time utilizing campus/digital library study spaces (yours: {vals['library_hours_per_week']:g} hrs/week).")
    if vals['tutoring_participation'] == 0 and pred in ["High", "Medium"]:
        recs.append("Participate in departmental academic tutoring sessions to strengthen foundational topics.")
    if status['discussion_participation_freq'] == 'bad':
        recs.append(f"Increase academic discussion and study group participation (yours: {vals['discussion_participation_freq']:g}/wk, average: {avg['discussion_participation_freq']:.1f}/wk).")
    if status['exercise_hours_per_week'] == 'bad':
        recs.append(f"Maintain routine physical exercise for improved cognitive focus and stress relief (yours: {vals['exercise_hours_per_week']:g} hrs/week).")

    if pred == "Low":
        base = [
            "Maintain current outstanding attendance, study habits, and self-learning routines.",
            "Continue regular assignment submission and academic engagement."
        ]
        if vals['internship_experience'] == 0:
            base.append("Explore internship and practical opportunities to apply theoretical knowledge in industry.")
        if vals['competition_participation'] == 0:
            base.append("Participate in subject-related academic competitions, hackathons, and symposiums to build advanced skills.")
        return base + ([f"Consider fine-tuning: {r}" for r in recs] if recs else [])
    
    if pred == "Medium":
        if not recs:
            recs.append("Stay consistent across all study modules; incremental weekly habit gains will move you into Low risk.")
        return recs

    # High Risk
    recs.append("Schedule an urgent academic advising consultation with faculty mentors.")
    return recs


def get_model_metrics(model, df, feature_cols):
    """Evaluates the saved model on the test split. Returns None if anything does not match."""
    try:
        X = df[feature_cols]
        y = df['risk_label']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        preds = model.predict(X_test)
        return {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, preds, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_test, preds, average='weighted', zero_division=0),
            'n_test': len(X_test),
            'n_train': len(df) - len(X_test),
        }
    except Exception:
        return None