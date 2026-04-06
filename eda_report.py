"""Tier 2 — Automated EDA Report Generator

A reusable module that accepts any DataFrame and automatically generates:
- Data profile (shape, types, missing values)
- Distribution plots for all numeric columns
- Correlation heatmap
- Missing data visualization
- Outlier summary (IQR method)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDAReport:
    def __init__(self, df, output_dir="output/eda_report", plot_style="whitegrid"):
        self.df = df
        self.output_dir = output_dir
        self.plot_style = plot_style
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_style(self.plot_style)

    def generate_profile(self):
        """Data profile (shape, types, missing values)"""
        profile_path = os.path.join(self.output_dir, "profile.txt")
        with open(profile_path, "w") as f:
            f.write("Data Profile Report\n")
            f.write("===================\n\n")
            f.write(f"Shape: {self.df.shape}\n\n")
            f.write("Data Types:\n")
            f.write(self.df.dtypes.to_string())
            f.write("\n\nMissing Values:\n")
            f.write(self.df.isnull().sum().to_string())
        return profile_path

    def plot_distributions(self, columns=None):
        """Distribution plots for all numeric columns or specified columns."""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        dist_dir = os.path.join(self.output_dir, "distributions")
        os.makedirs(dist_dir, exist_ok=True)
        
        for col in columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.savefig(os.path.join(dist_dir, f"{col}_dist.png"))
            plt.close()

    def plot_correlation_heatmap(self):
        """Correlation heatmap"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"))
        plt.close()

    def plot_missing_data(self):
        """Missing data visualization"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.df.columns, y=self.df.isnull().sum())
        plt.title("Missing Values per Column")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.savefig(os.path.join(self.output_dir, "missing_data.png"))
        plt.close()

    def generate_outlier_summary(self):
        """Outlier summary (IQR method)"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        summary_path = os.path.join(self.output_dir, "outlier_summary.txt")
        
        with open(summary_path, "w") as f:
            f.write("Outlier Summary (IQR Method)\n")
            f.write("============================\n\n")
            for col in numeric_df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                f.write(f"{col}:\n")
                f.write(f"  Lower Bound: {lower_bound:.2f}\n")
                f.write(f"  Upper Bound: {upper_bound:.2f}\n")
                f.write(f"  Outlier Count: {len(outliers)}\n\n")
        return summary_path

    def run_full_report(self, columns=None):
        """Run all report generation steps."""
        print(f"Generating EDA report in {self.output_dir}...")
        self.generate_profile()
        self.plot_distributions(columns=columns)
        self.plot_correlation_heatmap()
        self.plot_missing_data()
        self.generate_outlier_summary()
        print("EDA report generated successfully.")

def run_tests():
    """Test the EDAReport module with varying DataFrames."""
    print("Running tests for EDAReport...")
    
    # Test case 1: Standard DataFrame
    df1 = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randint(0, 10, 100),
        'C': ['cat', 'dog'] * 50
    })
    report1 = EDAReport(df1, output_dir="output/test_report_1")
    report1.run_full_report()

    # Test case 2: Missing data patterns
    df2 = pd.DataFrame({
        'X': [1, 2, np.nan, 4, 5],
        'Y': [np.nan, np.nan, 3, 4, 5]
    })
    report2 = EDAReport(df2, output_dir="output/test_report_2")
    report2.run_full_report()
    
    print("Tests completed.")

if __name__ == "__main__":
    run_tests()
