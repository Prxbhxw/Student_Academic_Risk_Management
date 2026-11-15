import numpy as np
import matplotlib.pyplot as plt

CATEGORIES = ["Low", "Medium", "High"]
DONUT_COLORS = ['#2ecc71', '#f39c12', '#e74c3c']


def _draw_pair(ax, labels, s_vals, a_vals, ymax, subtitle, show_legend=False):
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, s_vals, w, label='Student', color='#3498db')
    ax.bar(x + w / 2, a_vals, w, label='Dataset Average', color='#95a5a6')
    for i in range(len(labels)):
        ax.text(x[i] - w / 2, s_vals[i] + ymax * 0.02, f"{s_vals[i]:g}", ha='center', fontsize=8)
        ax.text(x[i] + w / 2, a_vals[i] + ymax * 0.02, f"{a_vals[i]:.1f}", ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15 if len(labels) >= 4 else 0)
    ax.set_ylim(0, ymax)
    ax.set_title(subtitle, fontsize=9)
    if show_legend:
        ax.legend(fontsize=8)


def _two_panel(title, left, right):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8),
                                   gridspec_kw={'width_ratios': [len(left[0]), len(right[0])]})
    _draw_pair(ax1, *left, show_legend=True)
    _draw_pair(ax2, *right)
    fig.suptitle(title, fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def make_academic_chart(vals, avg):
    """Compares core academic metrics and backlogs/failed courses."""
    return _two_panel(
        "Academic Performance & Course History Comparison (Tier 1 & Tier 2 Pillars)",
        (['Attendance %', 'Test Score', 'Assignments %', 'Learning Plan %'],
         [vals['attendance_percentage'], vals['avg_test_score'], vals['assignments_submitted_pct'], vals['learning_plan_completion_pct']],
         [avg['attendance_percentage'], avg['avg_test_score'], avg['assignments_submitted_pct'], avg['learning_plan_completion_pct']],
         110, "Percentage / Score Scale (0-100)"),
        (['GPA', 'Backlogs', 'Failed Courses'],
         [vals['previous_semester_gpa'], vals['backlogs'], vals['prev_failed_courses']],
         [avg['previous_semester_gpa'], avg['backlogs'], avg['prev_failed_courses']],
         11, "GPA (0-10) & Course Counts"),
    )


def make_engagement_chart(vals, avg):
    """Compares study hours, digital learning, and practical quantified engagement."""
    hours_max = max(vals['study_hours_per_week'], vals['weekly_self_learning_hours'],
                    vals['online_learning_hours'], vals['library_hours_per_week'],
                    vals['extracurricular_participation'],
                    avg['study_hours_per_week'], avg['weekly_self_learning_hours'],
                    avg['online_learning_hours'], avg['library_hours_per_week'],
                    avg['extracurricular_participation'], 15) * 1.25
    return _two_panel(
        "Quantified Study, Mentorship & Practical Skills",
        (['Study Hrs', 'Self-Learn', 'Online Lrn', 'Library', 'Extra-Curr Hrs'],
         [vals['study_hours_per_week'], vals['weekly_self_learning_hours'],
          vals['online_learning_hours'], vals['library_hours_per_week'], vals['extracurricular_participation']],
         [avg['study_hours_per_week'], avg['weekly_self_learning_hours'],
          avg['online_learning_hours'], avg['library_hours_per_week'], avg['extracurricular_participation']],
         hours_max, "Weekly Time Allocation (Hours/Week)"),
        (['Intern Months', 'Tutoring/Mo', 'Competitions', 'Practical Opp', 'Discussions/Wk'],
         [vals['internship_experience'], vals['tutoring_participation'],
          vals['competition_participation'], vals['practical_opportunity_freq'], vals['discussion_participation_freq']],
         [avg['internship_experience'], avg['tutoring_participation'],
          avg['competition_participation'], avg['practical_opportunity_freq'], avg['discussion_participation_freq']],
         max(vals['internship_experience'], vals['tutoring_participation'], 12) * 1.2,
         "Quantified Practical & Mentorship Metrics"),
    )


def make_donut(probs, prediction, confidence):
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.pie([probs.get(c, 0) for c in CATEGORIES], colors=DONUT_COLORS, startangle=90,
           counterclock=False, wedgeprops=dict(width=0.38, edgecolor='white'))
    ax.text(0, 0, f"{prediction}\n{confidence:.0f}%", ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax.legend(CATEGORIES, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=8, frameon=False)
    return fig
