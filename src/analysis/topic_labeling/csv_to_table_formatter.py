#!/usr/bin/env python3
"""
CSV to Table Formatter
Converts keyword frequency CSV data to hierarchical table format
MSc Data Science Thesis - Perceptions of Maths on YouTube
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

class CSVTableFormatter:
    def __init__(self):
        """Initialize CSV to table formatter."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        
    def load_keyword_frequencies_csv(self, csv_path):
        """Load the keyword frequencies CSV file."""
        print(f"Loading CSV data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows of data")
        return df
    
    def parse_keyword_frequencies(self, keyword_freq_string):
        """Parse the JSON string of keyword frequencies."""
        try:
            return json.loads(keyword_freq_string)
        except:
            return {}
    
    def aggregate_keywords_by_topic(self, df):
        """Aggregate keyword frequencies by topic number."""
        topic_data = defaultdict(lambda: {
            'queries': set(),
            'total_documents': 0,
            'keyword_frequencies': defaultdict(int),
            'keyword_query_count': defaultdict(int)
        })
        
        # Process each row
        for _, row in df.iterrows():
            topic = row['topic_number']
            query = row['query']
            doc_count = row['actual_doc_count']
            
            # Parse keyword frequencies
            keyword_freqs = self.parse_keyword_frequencies(row['keyword_frequencies'])
            
            # Update topic data
            topic_data[topic]['queries'].add(query)
            topic_data[topic]['total_documents'] += doc_count
            
            # Aggregate keyword frequencies
            for keyword, freq in keyword_freqs.items():
                topic_data[topic]['keyword_frequencies'][keyword] += freq
                topic_data[topic]['keyword_query_count'][keyword] += 1
        
        return topic_data
    
    def format_topic_summary(self, topic_data, top_keywords=10):
        """Format topic data into hierarchical summary format."""
        summary_lines = [
            "KEYWORD FREQUENCY ANALYSIS SUMMARY",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            ""
        ]
        
        # Sort topics by number
        sorted_topics = sorted(topic_data.keys())
        
        for topic_num in sorted_topics:
            data = topic_data[topic_num]
            query_count = len(data['queries'])
            doc_count = data['total_documents']
            
            summary_lines.extend([
                f"TOPIC {topic_num}:",
                f"  Queries: {query_count}",
                f"  Documents: {doc_count:,}",
                f"  Common Keywords:"
            ])
            
            # Get top keywords by frequency
            top_keywords_by_freq = sorted(
                data['keyword_frequencies'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_keywords]
            
            # Format each keyword with query appearance percentage
            for keyword, total_freq in top_keywords_by_freq:
                query_appearances = data['keyword_query_count'][keyword]
                percentage = (query_appearances / query_count * 100) if query_count > 0 else 0
                
                summary_lines.append(
                    f"    • {keyword}: {total_freq} occurrences in {query_appearances} queries ({percentage:.0f}%)"
                )
            
            summary_lines.append("")  # Empty line between topics
        
        return summary_lines
    
    def save_formatted_summary(self, summary_lines, output_path):
        """Save the formatted summary to a text file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"Formatted summary saved to: {output_path}")
        return output_path
    
    def create_detailed_keyword_report(self, topic_data):
        """Create a detailed keyword report with additional statistics."""
        report_lines = [
            "DETAILED KEYWORD FREQUENCY REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]
        
        # Overall statistics
        total_topics = len(topic_data)
        total_queries = sum(len(data['queries']) for data in topic_data.values())
        total_documents = sum(data['total_documents'] for data in topic_data.values())
        
        report_lines.extend([
            "OVERALL STATISTICS:",
            f"  Total Topics: {total_topics}",
            f"  Total Queries: {total_queries}",
            f"  Total Documents: {total_documents:,}",
            ""
        ])
        
        # Aggregate all keywords across topics
        all_keywords = defaultdict(lambda: {'total_freq': 0, 'topic_appearances': 0})
        
        for topic_num, data in topic_data.items():
            for keyword, freq in data['keyword_frequencies'].items():
                all_keywords[keyword]['total_freq'] += freq
                all_keywords[keyword]['topic_appearances'] += 1
        
        # Top keywords overall
        top_overall = sorted(all_keywords.items(), key=lambda x: x[1]['total_freq'], reverse=True)[:20]
        
        report_lines.extend([
            "TOP 20 KEYWORDS ACROSS ALL TOPICS:",
            "-" * 50
        ])
        
        for keyword, stats in top_overall:
            report_lines.append(
                f"  {keyword:<20} {stats['total_freq']:>6} occurrences in {stats['topic_appearances']:>2} topics"
            )
        
        report_lines.append("")
        
        return report_lines
    
    def process_csv_to_table(self, csv_path, output_dir=None):
        """Main processing function to convert CSV to table format."""
        print("=" * 60)
        print("CSV TO TABLE FORMATTER")
        print("=" * 60)
        
        # Set output directory
        if output_dir is None:
            output_dir = self.base_path / "results" / "analysis" / "formatted_summaries"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and process data
        df = self.load_keyword_frequencies_csv(csv_path)
        topic_data = self.aggregate_keywords_by_topic(df)
        
        # Create formatted summary (similar to hierarchical format)
        summary_lines = self.format_topic_summary(topic_data)
        summary_path = output_dir / f"keyword_frequency_summary_{self.timestamp}.txt"
        self.save_formatted_summary(summary_lines, summary_path)
        
        # Create detailed report
        report_lines = self.create_detailed_keyword_report(topic_data)
        combined_lines = report_lines + [""] + summary_lines
        report_path = output_dir / f"keyword_frequency_detailed_report_{self.timestamp}.txt"
        self.save_formatted_summary(combined_lines, report_path)
        
        print(f"\nFiles generated:")
        print(f"  • Summary: {summary_path.name}")
        print(f"  • Detailed Report: {report_path.name}")
        print(f"  • Output directory: {output_dir}")
        
        return summary_path, report_path

def main():
    """Main execution function."""
    formatter = CSVTableFormatter()
    
    # Default CSV path (update if needed)
    csv_path = "/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/results/analysis/topic_model/topic_keyword_frequencies/keyword_frequencies_detailed_20250816_220750.csv"
    
    # Process the CSV
    summary_path, report_path = formatter.process_csv_to_table(csv_path)
    
    print("\n" + "=" * 60)
    print("FORMATTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()