"""Lab 4 — Descriptive Analytics: Student Performance EDA

Conduct exploratory data analysis on the student performance dataset.
Produce distribution plots, correlation analysis, hypothesis tests,
and a written findings report.

Usage:
    python eda_analysis.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_profile(filepath):
    """Load the dataset and generate a data profile report.

    Args:
        filepath: path to the CSV file (e.g., 'data/student_performance.csv')

    Returns:
        DataFrame: the loaded dataset

    Side effects:
        Saves a text profile to output/data_profile.txt containing:
        - Shape (rows, columns)
        - Data types for each column
        - Missing value counts per column
        - Descriptive statistics for numeric columns
    """
    df = pd.read_csv(filepath)

    with open("output/data_profile.txt", "w") as f:
        f.write("Data Profile Report\n")
        f.write("===================\n\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Data Types:\n")
        f.write(df.dtypes.to_string())
        f.write("\n\nMissing Values per Column:\n")
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_report = pd.DataFrame({
            'Count': missing_count,
            'Percentage': missing_pct
        })
        f.write(missing_report.to_string())
        f.write("\n\nHandling Decisions:\n")
        f.write("- commute_minutes: Imputed missing values with median. Reasoning: Missing values are ~9% and likely MCAR; median is robust to outliers.\n")
        f.write("- scholarship: Imputed missing values with 'None'. Reasoning: Scholarship status is missing for ~19%; assuming missing means no scholarship.\n")
        f.write("- study_hours_weekly: No missing values found in current dataset, but if any were present, they would be dropped to ensure data quality.\n")
        f.write("\nDescriptive Statistics:\n")
        f.write(df.describe().to_string())

    # Handle missing values
    df['commute_minutes'] = df['commute_minutes'].fillna(df['commute_minutes'].median())
    df['scholarship'] = df['scholarship'].fillna('None')
    df = df.dropna(subset=['study_hours_weekly'])

    return df


def plot_distributions(df):
    """Create distribution plots for key numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least 3 distribution plots (histograms with KDE or box plots)
        as PNG files in the output/ directory. Each plot should have a
        descriptive title that states what the distribution reveals.
    """
    # Distribution of GPA
    plt.figure(figsize=(10, 6))
    sns.histplot(df['gpa'], kde=True, color='skyblue')
    plt.title('Distribution of GPA (Left-skewed)')
    plt.xlabel('GPA')
    plt.ylabel('Frequency')
    plt.savefig('output/gpa_distribution.png')
    plt.close()

    # Distribution of Study Hours
    plt.figure(figsize=(10, 6))
    sns.histplot(df['study_hours_weekly'], kde=True, color='green')
    plt.title('Distribution of Weekly Study Hours')
    plt.xlabel('Weekly Study Hours')
    plt.ylabel('Frequency')
    plt.savefig('output/study_hours_distribution.png')
    plt.close()

    # Distribution of Attendance
    plt.figure(figsize=(10, 6))
    sns.histplot(df['attendance_pct'], kde=True, color='orange')
    plt.title('Distribution of Attendance Percentage')
    plt.xlabel('Attendance Percentage')
    plt.ylabel('Frequency')
    plt.savefig('output/attendance_distribution.png')
    plt.close()

    # GPA by Department (Boxplot)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='department', y='gpa', data=df)
    plt.title('GPA Distribution across Departments (Boxplot)')
    plt.xlabel('Department')
    plt.ylabel('GPA')
    plt.savefig('output/gpa_by_department_boxplot.png')
    plt.close()

    # GPA by Department (Violin plot)
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='department', y='gpa', data=df, inner="quartile")
    plt.title('GPA Distribution across Departments (Violin Plot)')
    plt.xlabel('Department')
    plt.ylabel('GPA')
    plt.savefig('output/gpa_by_department_violin.png')
    plt.close()

    # Scholarship distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='scholarship', data=df)
    plt.title('Distribution of Scholarships')
    plt.xlabel('Scholarship Type')
    plt.ylabel('Count')
    plt.savefig('output/scholarship_distribution.png')
    plt.close()


def plot_correlations(df):
    """Analyze and visualize relationships between numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least one correlation visualization to the output/ directory
        (e.g., a heatmap, scatter plot, or pair plot).
    """
    # Compute correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Student Metrics')
    plt.savefig('output/correlation_heatmap.png')
    plt.close()

    # Find the two most correlated pairs (excluding self-correlation)
    corr_unstacked = corr_matrix.unstack()
    # Filter out self-correlations and duplicates
    corr_filtered = corr_unstacked[corr_unstacked < 1.0].sort_values(ascending=False)
    # The pairs are symmetric, so the first two will be the same pair (e.g., A-B and B-A)
    # So we take the 1st and 3rd if they are different, or just unique pairs.
    unique_pairs = []
    seen_pairs = set()
    for (v1, v2), val in corr_filtered.items():
        if tuple(sorted((v1, v2))) not in seen_pairs:
            unique_pairs.append(((v1, v2), val))
            seen_pairs.add(tuple(sorted((v1, v2))))
        if len(unique_pairs) >= 2:
            break

    # Create scatter plots for the top 2 pairs
    for i, ((v1, v2), val) in enumerate(unique_pairs):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=v1, y=v2, data=df, alpha=0.5)
        plt.title(f'Scatter Plot: {v1} vs {v2} (Correlation: {val:.2f})')
        plt.xlabel(v1)
        plt.ylabel(v2)
        plt.savefig(f'output/top_correlation_{i+1}.png')
        plt.close()


def run_hypothesis_tests(df):
    """Run statistical tests to validate observed patterns.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        dict: test results with keys like 'internship_ttest', 'dept_chi2',
              each containing the test statistic and p-value

    Side effects:
        Prints test results to stdout with interpretation.
    """
    results = {}

    # Hypothesis 1: Internship vs GPA
    group_yes = df[df['has_internship'] == 'Yes']['gpa']
    group_no = df[df['has_internship'] == 'No']['gpa']

    t_stat, p_val = stats.ttest_ind(group_yes, group_no)
    
    # Cohen's d
    n1, n2 = len(group_yes), len(group_no)
    var1, var2 = group_yes.var(), group_no.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohen_d = (group_yes.mean() - group_no.mean()) / pooled_std

    results['internship_ttest'] = {
        't_statistic': t_stat,
        'p_value': p_val,
        'cohen_d': cohen_d
    }

    print("\nHypothesis 1: Students with internships have a higher GPA than students without internships.")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    print(f"Cohen's d: {cohen_d:.4f}")
    if p_val < 0.05:
        print("Interpretation: There is a statistically significant difference in GPA between students with and without internships.")
    else:
        print("Interpretation: There is no statistically significant difference in GPA between students with and without internships.")

    # Hypothesis 2: Scholarship vs Department (Chi-square)
    contingency_table = pd.crosstab(df['scholarship'], df['department'])
    chi2, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)

    results['dept_chi2'] = {
        'chi2_statistic': chi2,
        'p_value': p_val_chi2,
        'dof': dof
    }

    print("\nHypothesis 2: Scholarship status is associated with department.")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_val_chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    if p_val_chi2 < 0.05:
        print("Interpretation: There is a statistically significant association between scholarship status and department.")
    else:
        print("Interpretation: There is no statistically significant association between scholarship status and department.")

    # Hypothesis 3: GPA differs across departments (ANOVA)
    departments = df['department'].unique()
    gpa_groups = [df[df['department'] == dept]['gpa'] for dept in departments]
    
    f_stat, p_val_anova = stats.f_oneway(*gpa_groups)
    results['dept_anova'] = {
        'f_statistic': f_stat,
        'p_value': p_val_anova
    }

    print("\nHypothesis 3: Average GPA differs across the five departments.")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_val_anova:.4f}")
    
    if p_val_anova < 0.05:
        print("Interpretation: There is a statistically significant difference in GPA across departments.")
        print("\nRunning post-hoc pairwise t-tests with Bonferroni correction...")
        
        posthoc_results = []
        for i in range(len(departments)):
            for j in range(i + 1, len(departments)):
                dept1, dept2 = departments[i], departments[j]
                group1 = df[df['department'] == dept1]['gpa']
                group2 = df[df['department'] == dept2]['gpa']
                t_stat_ph, p_val_ph = stats.ttest_ind(group1, group2)
                posthoc_results.append((dept1, dept2, t_stat_ph, p_val_ph))
        
        # Apply Bonferroni correction
        num_comparisons = len(posthoc_results)
        alpha_bonf = 0.05 / num_comparisons
        
        print(f"Bonferroni-corrected alpha: {alpha_bonf:.4f}")
        for dept1, dept2, t_stat_ph, p_val_ph in posthoc_results:
            if p_val_ph < alpha_bonf:
                print(f" - Significant difference between {dept1} and {dept2} (p={p_val_ph:.4f})")
    else:
        print("Interpretation: There is no statistically significant difference in GPA across departments.")

    return results


def main():
    """Orchestrate the full EDA pipeline."""
    os.makedirs("output", exist_ok=True)

    # Load and profile the dataset
    filepath = "data/student_performance.csv"
    df = load_and_profile(filepath)

    # Generate distribution plots
    plot_distributions(df)

    # Analyze correlations
    plot_correlations(df)

    # Run hypothesis tests
    run_hypothesis_tests(df)


if __name__ == "__main__":
    main()
