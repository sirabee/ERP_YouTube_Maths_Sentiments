#!/usr/bin/env python3
"""
CSV Data Processor for BERTopic Comparison Tables
Processes HDBSCAN and Variable K-means results to create professional comparison tables

INPUT FILES:
1. HDBSCAN results: bertopic_complete_pipeline_analysis_20250720_230249/_summary_reports/analysis_summary_stats.csv
2. Variable K-means results: optimised_variable_k_phase_4_20250722_224755/analysis_summary.csv

OUTPUT:
Professional comparison tables for all mathematical queries in multiple formats
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re
from datetime import datetime
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class BERTopicCSVProcessor:
    """
    Process CSV results from HDBSCAN and Variable K-means analyses
    Create professional comparison tables for MSc thesis
    """
    
    def __init__(self):
        self.setup_professional_styling()
        self.hdbscan_data = None
        self.kmeans_data = None
        self.output_dir = Path("comparison_tables_3")
        self.output_dir.mkdir(exist_ok=True)

    def setup_professional_styling(self):
        sns.set(style="whitegrid")
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16
        })

    def load_csvs(self, hdbscan_path, kmeans_path):
        self.hdbscan_data = pd.read_csv(hdbscan_path)
        self.kmeans_data = pd.read_csv(kmeans_path)
        self.hdbscan_data = self.hdbscan_data.round(3)
        self.kmeans_data = self.kmeans_data.round(3)

    def make_table_plot(self, df, title, note, filename):
        fig, ax = plt.subplots(figsize=(10, 2.75))
        ax.axis('off')

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Adjust layout to prevent overlap
        fig.subplots_adjust(top=0.8, bottom=0.2)
        fig.text(0.5, 0.93, title, ha='center', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.07, note, ha='center', fontsize=9, style='italic')

        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    def generate_all_tables(self):
        # This method should be populated with actual logic as in original script,
        # using self.make_table_plot(...) with the loaded data.
        pass

if __name__ == "__main__":
    processor = BERTopicCSVProcessor()
    processor.load_csvs(
        "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis/BERTopic HDBSCAN Per Query 20250720/bertopic_complete_pipeline_analysis_20250720_230249/_summary_reports/analysis_summary_stats.csv",
        "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis/optimised_variable_k_phase_4_20250722_224755/analysis_summary.csv"
    )
    processor.generate_all_tables()