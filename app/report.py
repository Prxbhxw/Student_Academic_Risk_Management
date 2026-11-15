import io
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

STATUS_PDF = {'pos': "Positive", 'neu': "Neutral", 'bad': "Needs Attention"}
CATEGORIES = ["Low", "Medium", "High"]


def build_pdf(name, usn, prediction, confidence, conf_text, probs, vals, status,
              extracurricular, recommendations, model_info, charts):
    """charts = list of matplotlib figures to put on the last page."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        page = {"fig": None, "y": 0.0}

        def new_page():
            if page["fig"] is not None:
                pdf.savefig(page["fig"])
                plt.close(page["fig"])
            page["fig"] = plt.figure(figsize=(8.27, 11.69))
            page["y"] = 0.95

        def add(text, size=9.5, bold=False, gap=0.021, indent=0.08, wrap=90, sub=""):
            text = str(text).replace("$", r"\$")
            for line_txt in (textwrap.wrap(text, width=wrap, subsequent_indent=sub) or [""]):
                if page["y"] < 0.06:
                    new_page()
                page["fig"].text(indent, page["y"], line_txt, fontsize=size,
                                 fontweight="bold" if bold else "normal", va="top")
                page["y"] -= gap

        def heading(text):
            page["y"] -= 0.008
            add(text, size=11.5, bold=True, gap=0.028)

        def line(label, col, unit=""):
            """One line: label, student's value and the status word."""
            status_word = STATUS_PDF.get(status.get(col, 'neu'), 'Neutral')
            add(f"{label}: {vals[col]:g}{unit}  -  {status_word}")

        new_page()
        add("STUDENT ACADEMIC RISK ASSESSMENT REPORT", size=16, bold=True, gap=0.045)

        heading("1. Profile Information")
        add(f"Name: {name}")
        add(f"USN / Identifier: {usn}")

        heading("2. Prediction Result & Confidence")
        add(f"Predicted Risk Level: {prediction.upper()} RISK")
        add(f"Model Confidence: {confidence:.1f}% ({conf_text} confidence)")

        heading("3. Risk Class Probabilities")
        for c in CATEGORIES:
            add(f"{c} Risk: {probs.get(c, 0)}%")

        heading("4. Academic Pillars (High Impact Tiers)")
        line("Attendance Percentage", 'attendance_percentage', "%")
        line("Average Test Score", 'avg_test_score', "/100")
        line("Previous Semester GPA", 'previous_semester_gpa', "/10")
        line("Active Backlogs Count", 'backlogs')
        line("Previous Semester Failed Courses", 'prev_failed_courses')
        line("Assignments Submitted", 'assignments_submitted_pct', "%")
        line("Personal Learning Plan Completion", 'learning_plan_completion_pct', "%")

        heading("5. Quantified Study Habits & Rigor")
        line("Weekly Study Hours", 'study_hours_per_week', " hrs/wk")
        line("Weekly Self-Learning Time", 'weekly_self_learning_hours', " hrs/wk")
        line("Online Learning Duration", 'online_learning_hours', " hrs/wk")
        line("Library Study Hours", 'library_hours_per_week', " hrs/wk")

        heading("6. Quantified Mentorship, Practical Experience & Engagement")
        line("Tutoring Sessions Attended", 'tutoring_participation', " sessions/month")
        line("Internship & Practical Experience", 'internship_experience', " months")
        line("Practical Opportunity Frequency", 'practical_opportunity_freq', " opps")
        line("Subject Competition Participation", 'competition_participation', " events")
        line("Extracurricular Engagement Duration", 'extracurricular_participation', " hrs/wk")
        line("Academic Discussion Frequency", 'discussion_participation_freq', " /wk")
        line("Exercise & Wellness Duration", 'exercise_hours_per_week', " hrs/wk")
        line("Active Clubs / Activities", 'num_activities')
        line("Achievements / Prizes Won", 'achievements')

        heading("7. Tailored Priority Action Plan")
        for r in recommendations:
            add("•  " + r, sub="     ")

        heading("8. Predictive Model Architecture & Metrics")
        add(f"Algorithm: {model_info['algorithm']}")
        add(f"Parameters evaluated: {model_info['n_features']} features   |   Classes: {model_info['n_classes']}")
        add(f"Cohort training dataset: {model_info['dataset_size']} records")
        m = model_info['metrics']
        if m:
            add(f"Test Accuracy: {m['Accuracy'] * 100:.1f}%   |   Precision: {m['Precision'] * 100:.1f}%   |   Recall: {m['Recall'] * 100:.1f}%   |   F1 Score: {m['F1 Score'] * 100:.1f}%")

        # Last page: charts
        new_page()
        add("Comparison Visualizations vs Cohort Averages", size=13, bold=True, gap=0.035)
        for chart_fig, bottom in zip(charts, (0.60, 0.22)):
            img_buf = io.BytesIO()
            chart_fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
            img_buf.seek(0)
            page["fig"].figimage(plt.imread(img_buf), xo=50, yo=int(bottom * 11.69 * 72), origin='upper')
        pdf.savefig(page["fig"])
        plt.close(page["fig"])

    buf.seek(0)
    return buf.getvalue()
