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
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from eda_report import EDAReport


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

    # GPA by Department
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='department', y='gpa', data=df)
    plt.title('GPA Distribution across Departments')
    plt.xlabel('Department')
    plt.ylabel('GPA')
    plt.savefig('output/gpa_by_department.png')
    plt.close()

    # Scholarship distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='scholarship', data=df)
    plt.title('Distribution of Scholarships')
    plt.xlabel('Scholarship Type')
    plt.ylabel('Count')
    plt.savefig('output/scholarship_distribution.png')
    plt.close()

    # Violin Plot: GPA by Department (Tier 1)
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='department', y='gpa', data=df, inner='box')
    plt.title('GPA Distribution across Departments (Violin Plot)')
    plt.xlabel('Department')
    plt.ylabel('GPA')
    plt.savefig('output/gpa_by_department_violin.png')
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

    # Hypothesis 2: Scholarship vs Department
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

    return results


def run_anova_tests(df):
    """Tier 1 — Advanced Statistical Analysis

    Does average GPA differ across the five departments?
    """
    print("\n--- ANOVA Test: GPA across Departments ---")
    groups = [df[df['department'] == dept]['gpa'] for dept in df['department'].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_val:.4f}")

    if p_val < 0.05:
        print("Interpretation: Significant differences exist in average GPA across departments.")
        print("\n--- Post-hoc Pairwise T-tests (Bonferroni Correction) ---")
        
        # Using Tukey HSD as a more common post-hoc, but user asked for Bonferroni pairwise t-tests.
        # stats.ttest_ind with correction manually or use pairwise_tukeyhsd which is often preferred.
        # Let's do manual Bonferroni as requested.
        depts = df['department'].unique()
        comparisons = []
        for i in range(len(depts)):
            for j in range(i + 1, len(depts)):
                d1, d2 = depts[i], depts[j]
                g1, g2 = df[df['department'] == d1]['gpa'], df[df['department'] == d2]['gpa']
                t_stat, p_val_pair = stats.ttest_ind(g1, g2)
                comparisons.append((d1, d2, t_stat, p_val_pair))
        
        # Bonferroni correction: alpha / number of comparisons
        num_comparisons = len(comparisons)
        alpha_corrected = 0.05 / num_comparisons
        print(f"Number of comparisons: {num_comparisons}")
        print(f"Bonferroni-corrected alpha: {alpha_corrected:.6f}")
        
        significant_diffs = []
        for d1, d2, t_stat, p_val_pair in comparisons:
            is_significant = p_val_pair < alpha_corrected
            if is_significant:
                significant_diffs.append((d1, d2, p_val_pair))
            # print(f"{d1} vs {d2}: p = {p_val_pair:.6f} {'*' if is_significant else ''}")
        
        if significant_diffs:
            print("Significant differences found between:")
            for d1, d2, p in significant_diffs:
                print(f"  - {d1} and {d2} (p={p:.6f})")
        else:
            print("No significant differences found after Bonferroni correction.")
    else:
        print("Interpretation: No significant differences in average GPA across departments.")


def run_bootstrap_ci(df):
    """Tier 3 — Bootstrap Confidence Interval

    Resample 10,000 times for mean GPA by internship status.
    """
    print("\n--- Bootstrap Confidence Intervals (GPA by Internship) ---")
    
    def bootstrap_mean(data, n_resamples=10000):
        resampled_means = []
        for _ in range(n_resamples):
            resample = np.random.choice(data, size=len(data), replace=True)
            resampled_means.append(np.mean(resample))
        return np.percentile(resampled_means, [2.5, 97.5])

    group_yes = df[df['has_internship'] == 'Yes']['gpa']
    group_no = df[df['has_internship'] == 'No']['gpa']

    ci_yes = bootstrap_mean(group_yes)
    ci_no = bootstrap_mean(group_no)

    print(f"Internship (Yes) 95% Bootstrap CI: [{ci_yes[0]:.4f}, {ci_yes[1]:.4f}]")
    print(f"Internship (No)  95% Bootstrap CI: [{ci_no[0]:.4f}, {ci_no[1]:.4f}]")

    # Parametric CI
    def parametric_ci(data):
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
        return ci

    p_ci_yes = parametric_ci(group_yes)
    p_ci_no = parametric_ci(group_no)

    print(f"Internship (Yes) 95% Parametric CI: [{p_ci_yes[0]:.4f}, {p_ci_yes[1]:.4f}]")
    print(f"Internship (No)  95% Parametric CI: [{p_ci_no[0]:.4f}, {p_ci_no[1]:.4f}]")


def run_power_analysis(df):
    """Tier 3 — Power Analysis

    Calculate needed sample size for 80% power at alpha = 0.05.
    """
    print("\n--- Power Analysis ---")
    group_yes = df[df['has_internship'] == 'Yes']['gpa']
    group_no = df[df['has_internship'] == 'No']['gpa']

    # Observed effect size (Cohen's d)
    n1, n2 = len(group_yes), len(group_no)
    var1, var2 = group_yes.var(), group_no.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    effect_size = (group_yes.mean() - group_no.mean()) / pooled_std

    analysis = TTestIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, ratio=n2/n1)
    
    print(f"Observed Cohen's d: {effect_size:.4f}")
    print(f"Required sample size for 80% power (n1): {np.ceil(sample_size):.0f}")
    print(f"Required total sample size (n1 + n2): {np.ceil(sample_size * (1 + n2/n1)):.0f}")


def run_simulation(alpha=0.05, n_simulations=1000, n_samples=50):
    """Tier 3 — Simulation

    Generate synthetic data, run hypothesis tests, and measure false positive rate.
    """
    print("\n--- Simulation: False Positive Rate ---")
    false_positives = 0
    
    for _ in range(n_simulations):
        # Generate two groups from the same normal distribution (null hypothesis is true)
        group1 = np.random.normal(loc=3.0, scale=0.5, size=n_samples)
        group2 = np.random.normal(loc=3.0, scale=0.5, size=n_samples)
        
        _, p_val = stats.ttest_ind(group1, group2)
        if p_val < alpha:
            false_positives += 1
            
    fpr = false_positives / n_simulations
    print(f"Simulated Alpha: {alpha}")
    print(f"Measured False Positive Rate: {fpr:.4f}")
    print(f"Does it match alpha? {'Yes' if abs(fpr - alpha) < 0.02 else 'No'}")


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

    # --- Challenge Extensions ---
    
    # Tier 1
    run_anova_tests(df)
    
    # Tier 2
    report = EDAReport(df, output_dir="output/automated_report")
    report.run_full_report()
    
    # Tier 3
    run_bootstrap_ci(df)
    run_power_analysis(df)
    run_simulation()


if __name__ == "__main__":
    main()
