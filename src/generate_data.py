import pandas as pd
import numpy as np

np.random.seed(42)

NUM_STUDENTS = 1200

# 1. Existing Academic Features
student_ids = [f"S{1000 + i}" for i in range(NUM_STUDENTS)]
attendance = np.random.normal(loc=76, scale=14, size=NUM_STUDENTS).clip(30, 100)
avg_test_score = np.random.normal(loc=67, scale=16, size=NUM_STUDENTS).clip(15, 100)
assignments_submitted_pct = np.random.normal(loc=80, scale=15, size=NUM_STUDENTS).clip(20, 100)
study_hours_per_week = np.random.normal(loc=12, scale=5.5, size=NUM_STUDENTS).clip(0, 40)
previous_gpa = np.random.normal(loc=6.7, scale=1.4, size=NUM_STUDENTS).clip(1.5, 10.0)
extracurricular_participation = np.random.choice([0, 1], size=NUM_STUDENTS, p=[0.55, 0.45])
backlogs = np.random.poisson(lam=0.7, size=NUM_STUDENTS).clip(0, 5)
library_hours_per_week = np.random.normal(loc=6.5, scale=4.0, size=NUM_STUDENTS).clip(0, 20)
achievements = np.random.poisson(lam=0.8, size=NUM_STUDENTS).clip(0, 5)
num_activities = np.random.poisson(lam=1.3, size=NUM_STUDENTS).clip(0, 5)

# 2. 10 New Features Requested
# Previous Semester Failed Courses (count 0-5)
prev_failed_courses = np.random.poisson(lam=0.6, size=NUM_STUDENTS).clip(0, 5)

# Online Learning Duration (hours/week 0-30)
online_learning_hours = np.random.normal(loc=8.0, scale=4.5, size=NUM_STUDENTS).clip(0, 30)

# Academic Tutoring Participation (0 or 1)
tutoring_participation = np.random.choice([0, 1], size=NUM_STUDENTS, p=[0.60, 0.40])

# Subject-Related Competition Participation (count 0-5)
competition_participation = np.random.poisson(lam=0.9, size=NUM_STUDENTS).clip(0, 5)

# Internship and Practical Experience (0 or 1)
internship_experience = np.random.choice([0, 1], size=NUM_STUDENTS, p=[0.68, 0.32])

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

# Multi-factor academic wellness index (0 to 100 scale)
academic_index = (
    (attendance / 100.0) * 17.0 +
    (avg_test_score / 100.0) * 17.0 +
    (assignments_submitted_pct / 100.0) * 9.0 +
    (previous_gpa / 10.0) * 17.0 +
    (study_hours_per_week / 40.0) * 6.0 +
    (weekly_self_learning_hours / 35.0) * 6.0 +
    (learning_plan_completion_pct / 100.0) * 8.0 +
    (online_learning_hours / 30.0) * 4.0 +
    (library_hours_per_week / 20.0) * 4.0 +
    (tutoring_participation) * 3.0 +
    (competition_participation / 5.0) * 3.0 +
    (internship_experience) * 3.0 +
    (practical_opportunity_freq / 5.0) * 2.0 +
    (discussion_participation_freq / 10.0) * 3.0 +
    (achievements / 5.0) * 2.0 +
    (num_activities / 5.0) * 2.0 +
    (extracurricular_participation) * 2.0 +
    (exercise_hours_per_week / 14.0) * 2.0 -
    (backlogs / 5.0) * 12.0 -
    (prev_failed_courses / 5.0) * 14.0
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
    "extracurricular_participation": extracurricular_participation,
    "backlogs": backlogs,
    "library_hours_per_week": library_hours_per_week.round(1),
    "achievements": achievements,
    "num_activities": num_activities,
    # 10 New features requested
    "prev_failed_courses": prev_failed_courses,
    "online_learning_hours": online_learning_hours.round(1),
    "tutoring_participation": tutoring_participation,
    "competition_participation": competition_participation,
    "internship_experience": internship_experience,
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