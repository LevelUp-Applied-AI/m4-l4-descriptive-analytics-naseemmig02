"""Tests for the Tier 2 — Automated EDA Report Generator."""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pytest
from eda_report import EDAReportGenerator

@pytest.fixture
def sample_df():
    """Create a sample DataFrame with missing values and outliers."""
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.normal(10, 2, 100),
        'B': np.random.normal(50, 10, 100),
        'C': ['cat', 'dog', 'bird', 'fish'] * 25,
        'D': [1, 2, np.nan, 4] * 25
    })
    # Add outliers to column B
    df.loc[0, 'B'] = 150
    df.loc[1, 'B'] = -50
    return df

def test_profile_generation(sample_df, tmp_path):
    """Test data profile generation."""
    output_dir = tmp_path / "output_test"
    generator = EDAReportGenerator(sample_df, output_dir=str(output_dir))
    generator.generate_profile()
    
    profile_path = output_dir / "data_profile.txt"
    assert os.path.exists(profile_path)
    with open(profile_path, "r") as f:
        content = f.read()
        assert "Shape: (100, 4)" in content
        assert "Missing Values per Column" in content

def test_distribution_plots(sample_df, tmp_path):
    """Test distribution plot generation."""
    output_dir = tmp_path / "output_test"
    generator = EDAReportGenerator(sample_df, output_dir=str(output_dir))
    generator.plot_distributions(columns=['A', 'B'])
    
    assert os.path.exists(output_dir / "dist_A.png")
    assert os.path.exists(output_dir / "dist_B.png")

def test_correlation_heatmap(sample_df, tmp_path):
    """Test correlation heatmap generation."""
    output_dir = tmp_path / "output_test"
    generator = EDAReportGenerator(sample_df, output_dir=str(output_dir))
    generator.plot_correlation_heatmap()
    
    assert os.path.exists(output_dir / "correlation_heatmap.png")

def test_missing_data_map(sample_df, tmp_path):
    """Test missing data map generation."""
    output_dir = tmp_path / "output_test"
    generator = EDAReportGenerator(sample_df, output_dir=str(output_dir))
    generator.plot_missing_data()
    
    assert os.path.exists(output_dir / "missing_data_map.png")

def test_outlier_summary(sample_df, tmp_path):
    """Test outlier summary generation."""
    output_dir = tmp_path / "output_test"
    generator = EDAReportGenerator(sample_df, output_dir=str(output_dir))
    generator.outlier_summary(columns=['B'])
    
    summary_path = output_dir / "outlier_summary.txt"
    assert os.path.exists(summary_path)
    with open(summary_path, "r") as f:
            content = f.read()
            assert "Column: B" in content
            assert "Outlier Count: 3" in content
