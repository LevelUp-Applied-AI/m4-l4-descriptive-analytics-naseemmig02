"""Tier 3 — Statistical Simulation and Power Analysis

This module performs bootstrapping, power analysis, and statistical 
simulations to deepen the analytical understanding of the dataset.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

def bootstrap_ci(data, n_resamples=10000, confidence_level=0.95):
    """Compute the bootstrap confidence interval for the mean.

    Args:
        data (array-like): The data to resample.
        n_resamples (int): Number of bootstrap resamples.
        confidence_level (float): The confidence level (0 to 1).

    Returns:
        tuple: (lower_bound, upper_bound, resampled_means)
    """
    resampled_means = []
    for _ in range(n_resamples):
        resample = np.random.choice(data, size=len(data), replace=True)
        resampled_means.append(np.mean(resample))
    
    lower_pct = (1 - confidence_level) / 2 * 100
    upper_pct = (1 + confidence_level) / 2 * 100
    lower_bound = np.percentile(resampled_means, lower_pct)
    upper_bound = np.percentile(resampled_means, upper_pct)
    
    return lower_bound, upper_bound, resampled_means

def run_tier3_analysis(df):
    """Execute Tier 3: Bootstrap, Power Analysis, and Simulation."""
    os.makedirs("output/tier3", exist_ok=True)
    
    # 1. Bootstrap Confidence Interval for Mean GPA by Internship Status
    group_yes = df[df['has_internship'] == 'Yes']['gpa'].dropna()
    group_no = df[df['has_internship'] == 'No']['gpa'].dropna()
    
    lower_yes, upper_yes, means_yes = bootstrap_ci(group_yes)
    lower_no, upper_no, means_no = bootstrap_ci(group_no)
    
    # Parametric CI for comparison (t-distribution)
    def parametric_ci(data, confidence=0.95):
        mean = np.mean(data)
        std_err = stats.sem(data)
        h = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return mean - h, mean + h

    p_lower_yes, p_upper_yes = parametric_ci(group_yes)
    p_lower_no, p_upper_no = parametric_ci(group_no)
    
    print("\nTier 3: Bootstrap Confidence Intervals (95%) for Mean GPA")
    print(f"Internship (Yes): Bootstrap [{lower_yes:.4f}, {upper_yes:.4f}] vs Parametric [{p_lower_yes:.4f}, {p_upper_yes:.4f}]")
    print(f"Internship (No):  Bootstrap [{lower_no:.4f}, {upper_no:.4f}] vs Parametric [{p_lower_no:.4f}, {p_upper_no:.4f}]")

    # 2. Power Analysis
    # Calculate observed effect size (Cohen's d)
    n1, n2 = len(group_yes), len(group_no)
    var1, var2 = group_yes.var(), group_no.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohen_d = (group_yes.mean() - group_no.mean()) / pooled_std
    
    power_analysis = TTestIndPower()
    required_n = power_analysis.solve_power(effect_size=cohen_d, alpha=0.05, power=0.8, ratio=n2/n1)
    
    print(f"\nTier 3: Power Analysis")
    print(f"Observed Cohen's d: {cohen_d:.4f}")
    print(f"Required sample size (n1) for 80% power at alpha=0.05: {required_n:.2f}")

    # 3. Statistical Simulation: False Positive Rate
    # Null hypothesis is true: two groups from the same normal distribution
    n_simulations = 1000
    alpha = 0.05
    false_positives = 0
    
    for _ in range(n_simulations):
        # Generate two groups with same mean and std
        s1 = np.random.normal(loc=2.8, scale=0.4, size=100)
        s2 = np.random.normal(loc=2.8, scale=0.4, size=100)
        _, p_val = stats.ttest_ind(s1, s2)
        if p_val < alpha:
            false_positives += 1
            
    fpr = false_positives / n_simulations
    print(f"\nTier 3: Simulation - False Positive Rate")
    print(f"Simulations: {n_simulations}, Target Alpha: {alpha}")
    print(f"Observed False Positive Rate: {fpr:.4f}")
    
    # Save a plot of the bootstrap distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(means_yes, color='blue', label='Internship (Yes)', kde=True)
    sns.histplot(means_no, color='orange', label='Internship (No)', kde=True)
    plt.axvline(np.mean(group_yes), color='darkblue', linestyle='--', label='Mean (Yes)')
    plt.axvline(np.mean(group_no), color='darkorange', linestyle='--', label='Mean (No)')
    plt.title('Bootstrap Distributions of Mean GPA')
    plt.xlabel('Mean GPA')
    plt.legend()
    plt.savefig('output/tier3/bootstrap_gpa_means.png')
    plt.close()

if __name__ == "__main__":
    from eda_analysis import load_and_profile
    df = load_and_profile("data/student_performance.csv")
    run_tier3_analysis(df)
