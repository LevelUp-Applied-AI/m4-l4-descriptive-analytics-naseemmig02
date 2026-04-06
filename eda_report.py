"""Tier 2 — Automated EDA Report Generator

A reusable module to automatically generate data profiles, distributions, 
correlations, and outlier summaries for any pandas DataFrame.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

class EDAReportGenerator:
    def __init__(self, df, output_dir="output_auto", plot_style="whitegrid"):
        """Initialize the generator with a DataFrame and configuration.

        Args:
            df (pd.DataFrame): The dataset to analyze.
            output_dir (str): Directory to save reports and plots.
            plot_style (str): Seaborn plot style (e.g., 'whitegrid', 'dark', 'ticks').
        """
        self.df = df
        self.output_dir = output_dir
        self.plot_style = plot_style
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_style(self.plot_style)

    def generate_profile(self):
        """Generate a basic data profile (shape, types, missing values)."""
        profile_path = os.path.join(self.output_dir, "data_profile.txt")
        with open(profile_path, "w") as f:
            f.write("Automated Data Profile Report\n")
            f.write("=============================\n\n")
            f.write(f"Shape: {self.df.shape}\n\n")
            f.write("Data Types:\n")
            f.write(self.df.dtypes.to_string())
            f.write("\n\nMissing Values per Column:\n")
            missing_count = self.df.isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            missing_report = pd.DataFrame({
                'Count': missing_count,
                'Percentage': missing_pct
            })
            f.write(missing_report.to_string())
        print(f"Profile saved to {profile_path}")

    def plot_distributions(self, columns=None):
        """Plot distributions for numeric columns.

        Args:
            columns (list): List of columns to plot. If None, plots all numeric columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col in self.df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}')
                plt.savefig(os.path.join(self.output_dir, f'dist_{col}.png'))
                plt.close()
        print(f"Distribution plots saved to {self.output_dir}")

    def plot_correlation_heatmap(self):
        """Generate a heatmap of Pearson correlations for numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("No numeric columns found for correlation heatmap.")
            return

        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
        plt.close()
        print(f"Correlation heatmap saved to {self.output_dir}")

    def plot_missing_data(self):
        """Visualize missing data patterns using a heatmap."""
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Missing Data Map (Yellow indicates missing)')
        plt.savefig(os.path.join(self.output_dir, 'missing_data_map.png'))
        plt.close()
        print(f"Missing data map saved to {self.output_dir}")

    def outlier_summary(self, columns=None):
        """Identify outliers using the Interquartile Range (IQR) method.

        Args:
            columns (list): List of columns to check. If None, checks all numeric columns.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        summary_path = os.path.join(self.output_dir, "outlier_summary.txt")
        with open(summary_path, "w") as f:
            f.write("Outlier Summary (IQR Method)\n")
            f.write("============================\n\n")
            for col in columns:
                if col in self.df.columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                    f.write(f"Column: {col}\n")
                    f.write(f"  - Lower Bound: {lower_bound:.2f}\n")
                    f.write(f"  - Upper Bound: {upper_bound:.2f}\n")
                    f.write(f"  - Outlier Count: {len(outliers)}\n")
                    f.write(f"  - Outlier Percentage: {(len(outliers) / len(self.df)) * 100:.2f}%\n\n")
        print(f"Outlier summary saved to {summary_path}")

    def run_full_report(self, columns=None):
        """Run all analysis steps and generate the full report."""
        self.generate_profile()
        self.plot_distributions(columns)
        self.plot_correlation_heatmap()
        self.plot_missing_data()
        self.outlier_summary(columns)
        print(f"Full report generated in {self.output_dir}")

if __name__ == "__main__":
    # Example usage with the student performance data
    data_path = "data/student_performance.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        generator = EDAReportGenerator(df, output_dir="output_automated_report")
        generator.run_full_report()
    else:
        print(f"Data file not found at {data_path}")
