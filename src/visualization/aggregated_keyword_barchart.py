#!/usr/bin/env python3
"""
Aggregated Keyword Frequency Bar Chart with Singular/Plural Normalization
Creates a bar chart of most frequent keywords across all topics with normalization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
import re
from collections import defaultdict

class AggregatedKeywordVisualizer:
    def __init__(self):
        """Initialize aggregated keyword visualizer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "visualizations" / "aggregated_keywords"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for clean professional charts
        plt.style.use('default')
        
        print("Aggregated Keyword Visualizer Initialized")
        print(f"Output directory: {self.output_dir}")
    
    def normalize_singular_plural(self, word):
        """Normalize a word to its singular form using simple rules."""
        word_lower = word.lower()
        
        # Simple rule-based approach for common patterns
        if word_lower.endswith('s') and len(word_lower) > 1:
            # Check if removing 's' gives us a reasonable singular form
            candidate = word_lower[:-1]
            # Don't modify if it would create very short words or known exceptions
            if len(candidate) >= 3 and candidate not in ['thi', 'wa', 'i', 'clas', 'glas', 'les', 'plu']:
                return candidate
        elif word_lower.endswith('es') and len(word_lower) > 2:
            # Handle -es endings (classes -> class, boxes -> box)
            candidate = word_lower[:-2]
            if len(candidate) >= 3:
                return candidate
        elif word_lower.endswith('ies') and len(word_lower) > 3:
            # Handle -ies endings (studies -> study)
            return word_lower[:-3] + 'y'
        
        return word_lower
    
    def parse_summary_file(self, summary_path):
        """Parse the keyword frequency summary text file."""
        with open(summary_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        all_keywords = defaultdict(int)
        current_topic = None
        
        print("\nParsing topics and normalizing keywords...")
        normalization_log = defaultdict(list)
        
        for line in lines:
            line = line.strip()
            
            # Match topic header
            if line.startswith("TOPIC"):
                topic_match = re.match(r"TOPIC (\d+):", line)
                if topic_match:
                    current_topic = int(topic_match.group(1))
            
            # Match keyword entries
            elif line.startswith("•") and current_topic is not None:
                # Parse: • keyword: count occurrences in X queries (Y%)
                keyword_match = re.match(r"•\s*([^:]+):\s*(\d+)\s*occurrences", line)
                if keyword_match:
                    keyword = keyword_match.group(1).strip()
                    count = int(keyword_match.group(2))
                    
                    # Normalize the keyword
                    normalized = self.normalize_singular_plural(keyword)
                    
                    # Track normalization for reporting
                    if normalized != keyword.lower():
                        normalization_log[normalized].append((keyword, count))
                    
                    # Add to aggregated count
                    all_keywords[normalized] += count
        
        # Print normalization summary
        print("\nNormalization combinations found:")
        for normalized, originals in normalization_log.items():
            if len(originals) > 1:
                orig_str = ' + '.join([f"{orig}({cnt})" for orig, cnt in originals])
                total = sum(cnt for _, cnt in originals)
                print(f"  {orig_str} → {normalized} (total: {total})")
        
        return all_keywords, normalization_log
    
    def create_aggregated_barchart(self, all_keywords, n_words=15):
        """Create a bar chart of top aggregated keywords."""
        # Sort keywords by frequency
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:n_words]
        words = [kw[0] for kw in sorted_keywords]
        frequencies = [kw[1] for kw in sorted_keywords]
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        ax.set_facecolor('white')
        
        # Create horizontal bar chart with green color scheme
        y_pos = np.arange(len(words))
        
        # Use single green color similar to reference
        bars = ax.barh(y_pos, frequencies, color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=1.0)
        
        # Customize appearance
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=12, fontweight='bold')
        ax.invert_yaxis()  # Top word at the top
        ax.set_xlabel('Total Frequency Across All Topics', fontsize=13, fontweight='bold')
        ax.set_title('Top Aggregated Keywords Across All Topics', fontsize=16, fontweight='bold', pad=20)  
        
        # Calculate proper x-axis limits to create space for labels
        max_freq = max(frequencies)
        ax.set_xlim(0, max_freq * 1.15)  # Add 15% padding on the right
        
        # Add value labels on bars (removed rank numbers)
        for bar, freq in zip(bars, frequencies):
            # Add frequency label at the end of bar with proper spacing
            ax.text(bar.get_width() + max_freq * 0.02, bar.get_y() + bar.get_height()/2,
                   f'{freq:,}', ha='left', va='center', fontsize=11, fontweight='bold')
        
        # Remove grid for clean appearance
        ax.grid(False)
        
        # Add light black border around the plot area
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        filename = f'aggregated_keywords_normalized_{self.timestamp}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\n✓ Aggregated keyword bar chart saved: {filename}")
        return filepath
    
    def create_normalization_report(self, all_keywords, normalization_log):
        """Create a detailed report of the normalization process."""
        report_lines = [
            "AGGREGATED KEYWORD FREQUENCY REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "With Singular/Plural Normalization",
            "=" * 60,
            "",
            "NORMALIZATION SUMMARY:",
            "-" * 40
        ]
        
        # Count normalizations
        total_normalizations = sum(len(originals) for originals in normalization_log.values() if len(originals) > 1)
        report_lines.append(f"Total keyword combinations normalized: {total_normalizations}")
        report_lines.append("")
        
        # List all normalizations
        report_lines.append("NORMALIZATION DETAILS:")
        report_lines.append("-" * 40)
        
        for normalized, originals in sorted(normalization_log.items()):
            if len(originals) > 1:
                total_freq = sum(cnt for _, cnt in originals)
                report_lines.append(f"\n{normalized} (total: {total_freq:,})")
                for original, count in originals:
                    report_lines.append(f"  - {original}: {count:,}")
        
        report_lines.extend(["", "", "TOP 50 AGGREGATED KEYWORDS:", "-" * 40])
        
        # List top 50 keywords
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:50]
        for rank, (keyword, freq) in enumerate(sorted_keywords, 1):
            report_lines.append(f"{rank:3}. {keyword:<20} {freq:>8,} occurrences")
        
        # Save report
        report_path = self.output_dir / f'normalization_report_{self.timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ Normalization report saved: {report_path.name}")
        return report_path
    
    def generate_visualization(self, summary_path):
        """Main function to generate aggregated visualization."""
        print("\n" + "=" * 60)
        print("GENERATING AGGREGATED KEYWORD BAR CHART")
        print("=" * 60)
        
        # Parse and normalize keywords
        print(f"\nParsing summary file: {summary_path}")
        all_keywords, normalization_log = self.parse_summary_file(summary_path)
        
        print(f"\nTotal unique keywords after normalization: {len(all_keywords)}")
        print(f"Total keyword occurrences: {sum(all_keywords.values()):,}")
        
        # Generate bar chart
        print("\nGenerating aggregated bar chart...")
        chart_path = self.create_aggregated_barchart(all_keywords)
        
        # Generate normalization report
        print("\nGenerating normalization report...")
        report_path = self.create_normalization_report(all_keywords, normalization_log)
        
        print("\n" + "=" * 60)
        print("VISUALIZATION COMPLETE")
        print("=" * 60)
        print(f"\nOutput directory: {self.output_dir}")
        
        return chart_path, report_path

def main():
    """Main execution function."""
    visualizer = AggregatedKeywordVisualizer()
    
    # Default summary file path (corrected)
    summary_path = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/analysis/topic_model/formatted_summaries/keyword_frequency_summary_20250829_210917.txt"
    
    # Generate visualization
    chart_path, report_path = visualizer.generate_visualization(summary_path)
    
    print("\nGenerated files:")
    print(f"  • Bar chart: {chart_path.name}")
    print(f"  • Report: {report_path.name}")

if __name__ == "__main__":
    main()