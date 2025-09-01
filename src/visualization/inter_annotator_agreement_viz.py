#!/usr/bin/env python3
"""
Inter-Annotator Agreement Visualization
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class InterAnnotatorVisualizer:
    """Generate visualizations for inter-annotator agreement analysis."""
    
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.metrics = self.data['metrics']
        self.total = self.data['total_annotations']
        
        # Create output directory
        self.output_dir = Path(json_path).parent.parent.parent / "figures" / \
                         "inter_annotator" / f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(self, name, data):
        """Create confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        matrix = np.array(data['confusion_matrix'])
        labels = data['labels']
        
        # Create annotated heatmap
        sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=labels, yticklabels=labels,
                   square=True, cbar_kws={'label': 'Count'})
        
        # Add percentages
        for i in range(len(labels)):
            for j in range(len(labels)):
                pct = (matrix[i, j] / matrix.sum()) * 100
                ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                       ha='center', va='center', fontsize=8, color='darkblue')
        
        ax.set_xlabel('Annotator 2', fontweight='bold')
        ax.set_ylabel('Annotator 1', fontweight='bold')
        ax.set_title(f"{data['annotation_type']}\n"
                    f"Agreement: {data['agreement_percentage']}% | κ: {data['kappa']:.3f}",
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filepath = self.output_dir / f"confusion_{name}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    def plot_kappa_comparison(self):
        """Compare Cohen's kappa across annotation types."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract data
        df = pd.DataFrame([{
            'Type': data['annotation_type'],
            'Kappa': data['kappa'],
            'Agreement': data['agreement_percentage']
        } for data in self.metrics.values()])
        
        # Bar chart
        colors = ['#e74c3c' if k < 0.4 else '#f39c12' if k < 0.6 else '#27ae60' 
                 for k in df['Kappa']]
        bars = ax1.bar(df['Type'], df['Kappa'], color=colors, alpha=0.8)
        
        # Add values
        for bar, kappa in zip(bars, df['Kappa']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{kappa:.3f}', ha='center', va='bottom')
        
        # Reference lines
        for y, label, color in [(0.4, 'Fair', 'orange'), (0.6, 'Moderate', 'green')]:
            ax1.axhline(y=y, color=color, linestyle='--', alpha=0.5, label=label)
        
        ax1.set_ylabel('Cohen\'s Kappa', fontweight='bold')
        ax1.set_title('Cohen\'s Kappa by Type', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.legend()
        
        # Scatter plot
        ax2.scatter(df['Agreement'], df['Kappa'], s=200, c=colors, alpha=0.7)
        for _, row in df.iterrows():
            ax2.annotate(row['Type'], (row['Agreement'], row['Kappa']),
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('Agreement %', fontweight='bold')
        ax2.set_ylabel('Cohen\'s Kappa', fontweight='bold')
        ax2.set_title('Kappa vs Raw Agreement', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / "kappa_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    def plot_sentiment_details(self):
        """Detailed sentiment agreement analysis."""
        if 'sentiment' not in self.metrics:
            return None
        
        data = self.metrics['sentiment']
        matrix = np.array(data['confusion_matrix'])
        labels = data['labels']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Normalized matrix
        matrix_norm = matrix / matrix.sum(axis=1, keepdims=True)
        sns.heatmap(matrix_norm, annot=True, fmt='.1%', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title('Normalized Confusion Matrix', fontweight='bold')
        ax1.set_xlabel('Annotator 2')
        ax1.set_ylabel('Annotator 1')
        
        # Disagreement patterns
        disagree = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and matrix[i, j] > 0:
                    disagree.append({
                        'Pattern': f"{labels[i]}→{labels[j]}",
                        'Count': matrix[i, j]
                    })
        
        if disagree:
            df = pd.DataFrame(disagree).sort_values('Count')
            ax2.barh(df['Pattern'], df['Count'], color='coral')
            ax2.set_xlabel('Count')
            ax2.set_title('Disagreement Patterns', fontweight='bold')
        
        # Per-class agreement
        class_agree = [matrix[i, i] / matrix[i, :].sum() * 100 
                      for i in range(len(labels))]
        ax3.bar(labels, class_agree, color=['#e74c3c', '#95a5a6', '#27ae60'])
        ax3.set_ylabel('Agreement %')
        ax3.set_title('Per-Class Agreement', fontweight='bold')
        ax3.set_ylim(0, 105)
        
        # Summary text
        ax4.axis('off')
        summary = f"""
        SUMMARY
        
        Total: {self.total} annotations
        Agreement: {data['agreement_percentage']}%
        Cohen's κ: {data['kappa']:.3f}
        
        Class Distribution:
        • Negative: {matrix[0, :].sum()}
        • Neutral: {matrix[1, :].sum()}
        • Positive: {matrix[2, :].sum()}
        """
        ax4.text(0.1, 0.7, summary, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Sentiment Annotation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = self.output_dir / "sentiment_analysis.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    def generate_report(self, files):
        """Generate markdown report."""
        report_path = self.output_dir / "agreement_report.md"
        
        report = f"""# Inter-Annotator Agreement Analysis

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Annotations:** {self.total}

## Results Summary

| Type | Cohen's κ | Interpretation | Agreement % |
|------|-----------|----------------|-------------|
"""
        for data in self.metrics.values():
            report += f"| {data['annotation_type']} | {data['kappa']:.3f} | {data['interpretation']} | {data['agreement_percentage']}% |\n"
        
        report += "\n## Visualizations\n\n"
        for name, path in files.items():
            if path:
                report += f"### {name}\n![{name}]({path.name})\n\n"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path
    
    def create_all_visualizations(self):
        """Generate all visualizations."""
        print("Generating visualizations...")
        files = {}
        
        # Confusion matrices
        for name, data in self.metrics.items():
            files[f"Confusion Matrix - {data['annotation_type']}"] = \
                self.plot_confusion_matrix(name, data)
        
        # Comparisons
        files['Kappa Comparison'] = self.plot_kappa_comparison()
        files['Sentiment Analysis'] = self.plot_sentiment_details()
        
        # Report
        report_path = self.generate_report(files)
        
        print(f"Output: {self.output_dir}")
        print(f"Report: {report_path.name}")
        return self.output_dir


def main():
    json_path = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/analysis/annotation_evaluation/inter_annotator_agreement_20250813_220609.json"
    
    visualizer = InterAnnotatorVisualizer(json_path)
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    main()