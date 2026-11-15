import pandas as pd
import numpy as np

np.random.seed(42)

NUM_STUDENTS = 1200

# 1. Core Academic Features
student_ids = [f"S{1000 + i}" for i in range(NUM_STUDENTS)]
attendance = np.random.normal(loc=76, scale=14, size=NUM_STUDENTS).clip(30, 100)
avg_test_score = np.random.normal(loc=67, scale=16, size=NUM_STUDENTS).clip(15, 100)
assignments_submitted_pct = np.random.normal(loc=80, scale=15, size=NUM_STUDENTS).clip(20, 100)
study_hours_per_week = np.random.normal(loc=12, scale=5.5, size=NUM_STUDENTS).clip(0, 40)
previous_gpa = np.random.normal(loc=6.7, scale=1.4, size=NUM_STUDENTS).clip(1.5, 10.0)

# Previously binary (Yes/No), now quantified:
# Extracurricular Participation: hours per week spent on extracurriculars (0 to 15 hrs/wk)
extracurricular_participation = np.random.exponential(scale=3.5, size=NUM_STUDENTS).clip(0, 15)

backlogs = np.random.poisson(lam=0.7, size=NUM_STUDENTS).clip(0, 5)
library_hours_per_week = np.random.normal(loc=6.5, scale=4.0, size=NUM_STUDENTS).clip(0, 20)
achievements = np.random.poisson(lam=0.8, size=NUM_STUDENTS).clip(0, 5)
num_activities = np.random.poisson(lam=1.3, size=NUM_STUDENTS).clip(0, 5)

# Previous Semester Failed Courses (count 0-5)
prev_failed_courses = np.random.poisson(lam=0.6, size=NUM_STUDENTS).clip(0, 5)

# Online Learning Duration (hours/week 0-30)
online_learning_hours = np.random.normal(loc=8.0, scale=4.5, size=NUM_STUDENTS).clip(0, 30)

# Previously binary (Yes/No), now quantified:
# Tutoring Participation: tutoring / mentoring sessions attended per month (0 to 12 sessions/month)
tutoring_participation = np.random.poisson(lam=2.5, size=NUM_STUDENTS).clip(0, 12)

# Subject-Related Competition Participation (count 0-5)
competition_participation = np.random.poisson(lam=0.9, size=NUM_STUDENTS).clip(0, 5)

# Previously binary (Yes/No), now quantified:
# Internship Experience: completed internship / practical project duration in months (0 to 12 months)
internship_experience = np.random.choice([0, 1, 2, 3, 6, 12], size=NUM_STUDENTS, p=[0.40, 0.15, 0.20, 0.15, 0.07, 0.03])

# Personal Learning Plan Completion (% 0-100)
learning_plan_completion_pct = np.random.normal(loc=74, scale=16, size=NUM_STUDENTS).clip(15, 100)

# Weekly Self-Learning Time (hours/week 0-35)
weekly_self_learning_hours = np.random.normal(loc=11.5, scale=5.0, size=NUM_STUDENTS).clip(0, 35)

# Internship and Practical Opportunity Frequency (count 0-5)
practical_opportunity_freq = np.random.poisson(lam=1.2, size=NUM_STUDENTS).clip(0, 5)

# Academic Discussion Participation Frequency (sessions/week 0-10)
discussion_participation_freq = np.random.normal(loc=4.2, scale=2.2, size=NUM_STUDENTS).clip(0, 10)

# Exercise Duration and Frequency (hours/week 0-14)
exercise_hours_per_week = np.random.normal(loc=5.0, scale=2.5, size=NUM_STUDENTS).clip(0, 14)

# ==============================================================================
# HIERARCHICAL WEIGHTED DECISION-MAKING POWER INDEX:
# Each parameter holds an explicit, calibrated weight reflecting its real-world
# predictive power over academic failure vs excellence.
# ==============================================================================
# Tier 1 - Primary Academic Pillars (Highest Power: Total ~55 pts)
#   - attendance (18 pts)
#   - avg_test_score (18 pts)
#   - previous_gpa (19 pts)
#
# Tier 2 - Critical Red Flags (Negative Penalties: Total ~-28 pts)
#   - prev_failed_courses (-15 pts)
#   - backlogs (-13 pts)
#
# Tier 3 - Academic Rigor & Study Habits (Strong Power: Total ~26 pts)
#   - assignments_submitted_pct (9 pts)
#   - learning_plan_completion_pct (8 pts)
#   - study_hours_per_week (5 pts)
#   - weekly_self_learning_hours (4 pts)
#
# Tier 4 - Mentorship & Experience Multipliers (Moderate-High Power: Total ~18 pts)
#   - tutoring_participation (sessions/mo, 0-12): 5 pts
#   - internship_experience (duration in mo, 0-12): 5 pts
#   - competition_participation (count, 0-5): 4 pts
#   - practical_opportunity_freq (count, 0-5): 4 pts
#
# Tier 5 - Support & Engagement Channels (Moderate Power: Total ~16 pts)
#   - online_learning_hours (0-30): 3 pts
#   - library_hours_per_week (0-20): 3 pts
#   - discussion_participation_freq (0-10): 3 pts
#   - extracurricular_participation (hrs/wk, 0-15): 3 pts
#   - achievements (count, 0-5): 2 pts
#   - num_activities (count, 0-5): 1 pt
#   - exercise_hours_per_week (0-14): 1 pt
# ==============================================================================

academic_index = (
    # Tier 1: Primary Academic Performance (Decisive power)
    (attendance / 100.0) * 18.0 +
    (avg_test_score / 100.0) * 18.0 +
    (previous_gpa / 10.0) * 19.0 +

    # Tier 2: Severe Academic Hazards (Negative penalty power)
    - (prev_failed_courses / 5.0) * 15.0 +
    - (backlogs / 5.0) * 13.0 +

    # Tier 3: Rigor & Self-Regulation
    (assignments_submitted_pct / 100.0) * 9.0 +
    (learning_plan_completion_pct / 100.0) * 8.0 +
    (study_hours_per_week / 40.0) * 5.0 +
    (weekly_self_learning_hours / 35.0) * 4.0 +

    # Tier 4: Quantified Mentorship & Practical Skills (Every hour / month counts)
    (tutoring_participation / 12.0) * 5.0 +
    (internship_experience / 12.0) * 5.0 +
    (competition_participation / 5.0) * 4.0 +
    (practical_opportunity_freq / 5.0) * 4.0 +

    # Tier 5: Quantified Engagement & Well-Being
    (online_learning_hours / 30.0) * 3.0 +
    (library_hours_per_week / 20.0) * 3.0 +
    (discussion_participation_freq / 10.0) * 3.0 +
    (extracurricular_participation / 15.0) * 3.0 +
    (achievements / 5.0) * 2.0 +
    (num_activities / 5.0) * 1.0 +
    (exercise_hours_per_week / 14.0) * 1.0
)

# Natural cutoffs: 18th percentile and 55th percentile
high_cutoff = np.percentile(academic_index, 18)
low_cutoff = np.percentile(academic_index, 55)

risk_label = []
for s in academic_index:
    if s >= low_cutoff:
        risk_label.append("Low")
    elif s >= high_cutoff:
        risk_label.append("Medium")
    else:
        risk_label.append("High")

df = pd.DataFrame({
    "student_id": student_ids,
    # Core academic features
    "attendance_percentage": attendance.round(1),
    "avg_test_score": avg_test_score.round(1),
    "assignments_submitted_pct": assignments_submitted_pct.round(1),
    "study_hours_per_week": study_hours_per_week.round(1),
    "previous_semester_gpa": previous_gpa.round(2),
    "extracurricular_participation": extracurricular_participation.round(1),  # Hours/week
    "backlogs": backlogs,
    "library_hours_per_week": library_hours_per_week.round(1),
    "achievements": achievements,
    "num_activities": num_activities,
    # 10 Additional features, all quantitative
    "prev_failed_courses": prev_failed_courses,
    "online_learning_hours": online_learning_hours.round(1),
    "tutoring_participation": tutoring_participation,  # Sessions/month
    "competition_participation": competition_participation,
    "internship_experience": internship_experience,  # Duration in months
    "learning_plan_completion_pct": learning_plan_completion_pct.round(1),
    "weekly_self_learning_hours": weekly_self_learning_hours.round(1),
    "practical_opportunity_freq": practical_opportunity_freq,
    "discussion_participation_freq": discussion_participation_freq.round(1),
    "exercise_hours_per_week": exercise_hours_per_week.round(1),
    "risk_label": risk_label
})

df.to_csv("data/students_dataset.csv", index=False)
print("Dataset generated successfully:", df.shape)
print("Distribution of Risk Labels:")
print(df["risk_label"].value_counts())
