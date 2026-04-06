# Findings Report: Hashemite Technical University Student Performance Analysis

## Dataset Description
- **Shape**: (2000, 10)
- **Columns**: `student_id`, `department`, `semester`, `course_load`, `study_hours_weekly`, `gpa`, `attendance_pct`, `has_internship`, `commute_minutes`, `scholarship`.
- **Notable Data Quality Issues**:
    - `commute_minutes`: Missing for ~9% of records. Imputed with the median (25.0).
    - `scholarship`: Missing for ~19% of records. Imputed with 'None' as it's assumed missing means no scholarship.
    - `study_hours_weekly`: No missing values were found in the provided dataset.

## Key Distribution Findings
- **GPA**: The distribution is left-skewed, with most students clustering between 2.5 and 3.5. A tail extends toward lower GPAs.
- **Weekly Study Hours**: Shows a roughly normal distribution centered around 14.9 hours.
- **GPA by Department**: Box plots and violin plots reveal that GPA distributions are fairly consistent across departments, with slight variations in medians and outliers. The violin plots show similar shapes for all departments, suggesting no major differences in the spread of academic performance.
- **Scholarships**: The distribution is balanced across Merit, Athletic, Need-based, and Department scholarships, with a significant number of students having no scholarship (None).
- **See output/gpa_distribution.png, output/study_hours_distribution.png, output/gpa_by_department_boxplot.png, and output/gpa_by_department_violin.png**.

## Notable Correlations
- **GPA and Study Hours**: A moderate positive correlation (r = 0.64) exists between weekly study hours and GPA. This suggests that students who study more tend to have higher GPAs.
- **GPA and Attendance**: There is a very weak positive correlation (r = 0.04), indicating that attendance alone is not a strong predictor of GPA in this dataset.
- **Caveat**: Correlation does not imply causation. Other factors such as prior knowledge or learning efficiency may also influence both study hours and GPA.
- **See output/correlation_heatmap.png and output/top_correlation_1.png**.

## Hypothesis Test Results

### Hypothesis 1: Students with internships have a higher GPA than students without internships.
- **Test Used**: Independent Samples T-test.
- **Results**:
    - T-statistic: 13.5644
    - P-value: 0.0000
    - Cohen's d: 0.69
- **Interpretation**: The result is statistically significant (p < 0.05). Students with internships have significantly higher GPAs than those without. The moderate effect size (d = 0.69) suggests this difference is practically meaningful.

### Hypothesis 2: Scholarship status is associated with department.
- **Test Used**: Chi-square Test of Independence.
- **Results**:
    - Chi-square statistic: 17.1358
    - P-value: 0.3769
    - Degrees of freedom: 16
- **Interpretation**: The result is not statistically significant (p > 0.05). There is no strong evidence of an association between a student's department and their scholarship status.

### Hypothesis 3: Average GPA differs across the five departments (ANOVA).
- **Test Used**: One-Way ANOVA.
- **Results**:
    - F-statistic: 0.6671
    - P-value: 0.6148
- **Interpretation**: The result is not statistically significant (p > 0.05). There is no strong evidence of a difference in mean GPA between any of the five departments.

## Challenge Extension Results

### Tier 1 — Advanced Statistical Analysis
- **ANOVA**: Confirmed no statistically significant difference in GPA across departments.
- **Violin Plots**: Provided deeper insight into GPA distribution shape by department, confirming consistency across the university (See `output/gpa_by_department_violin.png`).

### Tier 2 — Automated EDA Report Generator
- **Module**: Built a reusable `eda_report.py` that can handle any DataFrame.
- **Features**: Automatically generates data profiles, distribution plots, correlation heatmaps, missing data maps, and outlier summaries.
- **Verification**: Passed 5 automated tests verifying its robustness against different data patterns.

### Tier 3 — Statistical Simulation and Power Analysis
- **Bootstrap Confidence Intervals**: 
    - Internship (Yes): [2.95, 3.02] 
    - Internship (No): [2.68, 2.72]
    - The non-overlapping CIs further confirm the statistical significance found in the t-test.
- **Power Analysis**: To detect an effect size of d = 0.69 with 80% power at alpha = 0.05, a sample size of only ~23 students per group is required. Our current sample size (N=2000) provides extremely high statistical power.
- **Simulation**: A simulation of 1,000 tests under the null hypothesis yielded a false positive rate of ~5.3%, which aligns with our target alpha of 0.05.

## Actionable Recommendations
1. **Encourage Internship Participation**: Since students with internships show significantly higher GPAs, the university should expand its internship program and encourage more students to participate, as it may correlate with better academic outcomes.
2. **Promote Effective Study Habits**: Given the strong correlation between study hours and GPA, the university should provide workshops on effective study techniques to help students maximize the impact of their study time.
3. **Review Attendance Impact**: Since attendance shows a very weak correlation with GPA, the university might investigate whether class quality or instructional methods could be adjusted to make attendance a more impactful factor in student success.
