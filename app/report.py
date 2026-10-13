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

        heading("1. Student Information")
        add(f"Name: {name}")
        add(f"USN / Student ID: {usn}")

        heading("2. Prediction Result")
        add(f"Risk Level: {prediction.upper()} RISK")
        add(f"Prediction Confidence: {confidence:.1f}% ({conf_text} confidence)")

        heading("3. Risk Class Probabilities")
        for c in CATEGORIES:
            add(f"{c} Risk: {probs.get(c, 0)}%")

        heading("4. Academic Performance & Course History")
        line("Attendance", 'attendance_percentage', "%")
        line("Average Test Score", 'avg_test_score')
        line("Assignments Submitted", 'assignments_submitted_pct', "%")
        line("Previous Semester GPA", 'previous_semester_gpa')
        line("Active Backlogs", 'backlogs')
        line("Previous Semester Failed Courses", 'prev_failed_courses')

        heading("5. Study Habits & Self-Regulation")
        line("Weekly Study Hours", 'study_hours_per_week', " hrs/wk")
        line("Weekly Self-Learning Time", 'weekly_self_learning_hours', " hrs/wk")
        line("Online Learning Duration", 'online_learning_hours', " hrs/wk")
        line("Library Hours per Week", 'library_hours_per_week', " hrs/wk")
        line("Personal Learning Plan Completion", 'learning_plan_completion_pct', "%")

        heading("6. Practical Experience, Engagement & Well-Being")
        add(f"Academic Tutoring Participation: {'Yes' if vals['tutoring_participation'] else 'No'}")
        line("Subject-Related Competition Participation", 'competition_participation')
        add(f"Internship and Practical Experience: {'Yes' if vals['internship_experience'] else 'No'}")
        line("Practical Opportunity Frequency", 'practical_opportunity_freq')
        line("Academic Discussion Participation Frequency", 'discussion_participation_freq', " /wk")
        line("Exercise Duration and Frequency", 'exercise_hours_per_week', " hrs/wk")
        add(f"Extracurricular Participation: {extracurricular}")
        line("Activities / Clubs", 'num_activities')
        line("Achievements / Prizes Won", 'achievements')

        heading("7. Recommended Action Plan")
        for r in recommendations:
            add("•  " + r, sub="     ")

        heading("8. Predictive Model Architecture & Metrics")
        add(f"Algorithm: {model_info['algorithm']}")
        add(f"Parameters evaluated: {model_info['n_features']} features   |   Classes: {model_info['n_classes']}")
        add(f"Training dataset: {model_info['dataset_size']} student records")
        m = model_info['metrics']
        if m:
            add(f"Training/Test split: {m['n_train']} train / {m['n_test']} test students")
            add(f"Model Accuracy: {m['Accuracy'] * 100:.1f}%   |   Precision: {m['Precision'] * 100:.1f}%   |   Recall: {m['Recall'] * 100:.1f}%   |   F1 Score: {m['F1 Score'] * 100:.1f}%")

        # Last page: charts
        new_page()
        add("Comparison Visualizations", size=13, bold=True, gap=0.035)
        for chart_fig, bottom in zip(charts, (0.60, 0.22)):
            img_buf = io.BytesIO()
            chart_fig.savefig(img_buf, format="png", dpi=150)
            img_buf.seek(0)
            ax = page["fig"].add_axes([0.06, bottom, 0.88, 0.32])
            ax.imshow(plt.imread(img_buf, format="png"))
            ax.axis("off")

        pdf.savefig(page["fig"])
        plt.close(page["fig"])
    buf.seek(0)
    return buf