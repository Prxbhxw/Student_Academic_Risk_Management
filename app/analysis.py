from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# (column name, label shown on screen, type of factor, unit, decision power tier)
ANALYSIS_ITEMS = [
    ('avg_test_score', 'Test Score', 'main', '/100', 'Tier 1 - Highest Impact (18%)'),
    ('previous_semester_gpa', 'Previous Semester GPA', 'main', '/10', 'Tier 1 - Highest Impact (19%)'),
    ('attendance_percentage', 'Attendance Percentage', 'main', '%', 'Tier 1 - Highest Impact (18%)'),
    ('prev_failed_courses', 'Failed Courses (Prev Sem)', 'backlog', 'courses', 'Tier 2 - Severe Risk Penalty (-15%)'),
    ('backlogs', 'Active Backlogs', 'backlog', 'backlogs', 'Tier 2 - Severe Risk Penalty (-13%)'),
    ('learning_plan_completion_pct', 'Learning Plan Completion', 'main', '%', 'Tier 3 - Rigor & Habits (8%)'),
    ('assignments_submitted_pct', 'Assignment Submission', 'main', '%', 'Tier 3 - Rigor & Habits (9%)'),
    ('internship_experience', 'Internship & Practical Experience', 'engage', 'months', 'Tier 4 - Industry & Skills (5%)'),
    ('practical_opportunity_freq', 'Practical Opportunity Frequency', 'engage', 'opps', 'Tier 4 - Industry & Skills (4%)'),
    ('tutoring_participation', 'Tutoring Sessions Attended', 'engage', 'sessions/mo', 'Tier 4 - Mentorship & Support (5%)'),
    ('competition_participation', 'Subject Competitions', 'engage', 'events', 'Tier 4 - Advanced Skills (4%)'),
    ('extracurricular_participation', 'Extracurricular Engagement', 'engage', 'hrs/wk', 'Tier 5 - Holistic Growth (3%)'),
    ('discussion_participation_freq', 'Discussion Frequency', 'main', '/wk', 'Tier 5 - Engagement (3%)'),
    ('study_hours_per_week', 'Weekly Study Hours', 'main', 'hrs/wk', 'Tier 3 - Academic Rigor (5%)'),
    ('weekly_self_learning_hours', 'Self-Learning Hours', 'main', 'hrs/wk', 'Tier 3 - Academic Rigor (4%)'),
    ('achievements', 'Achievements / Prizes Won', 'engage', 'awards', 'Tier 5 - Growth (2%)'),
    ('library_hours_per_week', 'Library Study Hours', 'main', 'hrs/wk', 'Tier 5 - Engagement (3%)'),
    ('online_learning_hours', 'Online Learning Hours', 'main', 'hrs/wk', 'Tier 5 - Digital Study (3%)'),
    ('num_activities', 'Active Clubs & Societies', 'engage', 'clubs', 'Tier 5 - Engagement (1%)'),
    ('exercise_hours_per_week', 'Fitness & Exercise Hours', 'main', 'hrs/wk', 'Tier 5 - Wellness (1%)'),
]


def get_status(kind, s, a):
    """Returns 'pos', 'neu' or 'bad' by comparing the student's value with the average."""
    if kind == 'backlog':
        if s == 0:
            return 'pos'
        return 'neu' if s <= a else 'bad'
    if kind == 'engage':
        return 'pos' if s >= a else ('neu' if s > 0 else 'bad')
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
    for col, label, kind, unit, power in ANALYSIS_ITEMS:
        s, a = vals[col], avg[col]
        status[col] = get_status(kind, s, a)
        rows.append({
            'factor': label,
            'student': f"{s:g} {unit}".strip(),
            'average': f"{a:.1f} {unit}".strip(),
            'comparison': get_comparison(s, a),
            'status': status[col],
            'power': power
        })
    return status, rows


def confidence_label(p):
    if p >= 70:
        return "high"
    if p >= 50:
        return "moderate"
    return "low (borderline case)"


def get_recommendations(pred, vals, avg, status):
    """Recommendations are prioritized according to decision power tiers."""
    recs = []
    
    # Tier 1 & 2: Critical High-Impact Decisions
    if status['avg_test_score'] == 'bad':
        recs.append(f"🔴 [High Impact - 18% Power] Prioritize core test preparation: current average score is {vals['avg_test_score']:g} vs. cohort average of {avg['avg_test_score']:.1f}.")
    if status['previous_semester_gpa'] == 'bad':
        recs.append(f"🔴 [High Impact - 19% Power] GPA improvement plan required: current GPA is {vals['previous_semester_gpa']:.2f} vs. benchmark {avg['previous_semester_gpa']:.2f}.")
    if status['attendance_percentage'] == 'bad':
        recs.append(f"🔴 [High Impact - 18% Power] Attendance critical alert: attendance is at {vals['attendance_percentage']:g}% (cohort average {avg['attendance_percentage']:.1f}%).")
    if vals['prev_failed_courses'] > 0:
        recs.append(f"⚠️ [Severe Penalty - 15% Power] Enroll immediately in re-sit/recovery courses to clear {vals['prev_failed_courses']:g} failed course(s).")
    if vals['backlogs'] > 0:
        recs.append(f"⚠️ [Severe Penalty - 13% Power] Clear {vals['backlogs']:g} active backlog(s) before end-semester examinations.")

    # Tier 3: Rigor & Habits
    if status['assignments_submitted_pct'] == 'bad':
        recs.append(f"🟡 [Moderate Impact - 9% Power] Submit remaining coursework to increase assignment completion from {vals['assignments_submitted_pct']:g}% towards {avg['assignments_submitted_pct']:.1f}%.")
    if status['learning_plan_completion_pct'] == 'bad':
        recs.append(f"🟡 [Moderate Impact - 8% Power] Align study schedule with weekly syllabus to raise plan completion from {vals['learning_plan_completion_pct']:g}%.")
    if status['study_hours_per_week'] == 'bad':
        recs.append(f"🟡 [Moderate Impact - 5% Power] Increase weekly study time: currently {vals['study_hours_per_week']:g} hrs/wk vs. cohort average {avg['study_hours_per_week']:.1f} hrs/wk.")

    # Tier 4: Mentorship & Practical Skills (Quantitative boosts)
    if vals['tutoring_participation'] < avg['tutoring_participation']:
        recs.append(f"💡 [Mentorship Multiplier - 5% Power] Attend more academic tutoring sessions (currently {vals['tutoring_participation']:g} sessions/month vs cohort {avg['tutoring_participation']:.1f}/month).")
    if vals['internship_experience'] == 0:
        recs.append(f"💼 [Practical Experience - 5% Power] Gain applied industry experience or internships (each month provides strong resilience against academic risk).")
    if vals['competition_participation'] == 0:
        recs.append(f"🏆 [Skill Acceleration - 4% Power] Participate in academic hackathons or domain competitions to build hands-on mastery.")

    # Tier 5: Holistic Balance
    if vals['extracurricular_participation'] < 1.0:
        recs.append(f"🌱 [Holistic Balance - 3% Power] Allocate at least 2-4 hours/week for extracurriculars or technical clubs.")
    if status['exercise_hours_per_week'] == 'bad':
        recs.append(f"🏃 [Wellness Support - 1% Power] Maintain active physical routine ({vals['exercise_hours_per_week']:g} hrs/wk logged) for mental stamina.")

    if pred == "Low":
        base = [
            "⭐ Outstanding Profile: Core academic metrics (GPA, Test Scores, Attendance) and practical engagement are in the top tier.",
            "Continue participating in internships and advanced competitions to maintain industry readiness."
        ]
        return base + recs[:3]
    
    if pred == "Medium":
        if not recs:
            recs.append("Consistent progress noted: focusing on 2-3 extra hours of self-study and clearing pending assignments will elevate you to Low Risk.")
        return recs

    # High Risk
    recs.insert(0, "🚨 URGENT: Multiple Tier-1 academic parameters indicate high risk. Mandatory mentoring session required.")
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
