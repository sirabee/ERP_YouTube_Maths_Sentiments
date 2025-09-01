#!/usr/bin/env python3
"""
Simple Hierarchical Topic Analysis - Per Topic Number
Explores common keywords across queries for each topic number
MSc Data Science Thesis - Perceptions of Maths on YouTube
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

class SimpleHierarchicalAnalyzer:
    def __init__(self):
        """Initialize analyzer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "analysis" / "simple_hierarchical"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Simple Hierarchical Topic Analyzer")
        print(f"Output directory: {self.output_dir}")
    
    def load_data(self):
        """Load topic data."""
        topic_file = self.base_path / "results" / "models" / "bertopic_outputs" / "Optimised_Variable_K" / "merged_topic_info" / "merged_topic_info_sorted.csv"
        
        print(f"Loading data...")
        df = pd.read_csv(topic_file)
        print(f"Loaded {len(df)} topics from {df['query'].nunique()} queries")
        
        return df
    
    def extract_keywords(self, representation_str):
        """Extract keywords from representation string."""
        if pd.isna(representation_str):
            return []
        
        # Clean and split
        keywords = representation_str.replace('[', '').replace(']', '').replace("'", '').split(',')
        keywords = [k.strip().lower() for k in keywords[:10]]  # Top 10
        return keywords
    
    def analyze_topic(self, df, topic_num):
        """Analyze a specific topic number across queries."""
        print(f"\n--- Topic {topic_num} ---")
        
        # Filter for this topic
        topic_df = df[df['Topic'] == topic_num].copy()
        
        if len(topic_df) < 2:
            print(f"Topic {topic_num}: Only {len(topic_df)} query - skipping")
            return None
        
        print(f"Found {len(topic_df)} queries with Topic {topic_num}")
        print(f"Documents: {topic_df['Count'].sum():,}")
        
        # Count keywords across all queries
        keyword_counter = Counter()
        query_keywords = {}
        
        for _, row in topic_df.iterrows():
            keywords = self.extract_keywords(row['Representation'])
            query_keywords[row['query']] = keywords
            
            # Count each keyword
            for kw in keywords:
                keyword_counter[kw] += 1
        
        # Find common keywords (appear in multiple queries)
        common_keywords = []
        total_queries = len(topic_df)
        
        for keyword, count in keyword_counter.most_common(20):
            if count > 1:  # Appears in more than one query
                percentage = (count / total_queries) * 100
                common_keywords.append({
                    'keyword': keyword,
                    'count': count,
                    'percentage': percentage
                })
        
        # Create simple hierarchy if we have enough queries
        if len(topic_df) >= 3:
            self.create_simple_hierarchy(topic_df, topic_num)
        
        return {
            'topic_num': topic_num,
            'num_queries': len(topic_df),
            'documents': topic_df['Count'].sum(),
            'common_keywords': common_keywords,
            'queries': topic_df['query'].tolist()
        }
    
    def create_simple_hierarchy(self, topic_df, topic_num):
        """Create simple dendrogram for topic."""
        try:
            # Get all unique keywords
            all_keywords = set()
            query_labels = []
            
            for _, row in topic_df.iterrows():
                keywords = self.extract_keywords(row['Representation'])
                all_keywords.update(keywords)
                query_labels.append(row['query'][:20])  # Truncate long names
            
            keyword_list = sorted(list(all_keywords))
            
            # Create binary matrix (keyword present or not)
            matrix = []
            for _, row in topic_df.iterrows():
                keywords = self.extract_keywords(row['Representation'])
                vector = [1 if kw in keywords else 0 for kw in keyword_list]
                matrix.append(vector)
            
            matrix = np.array(matrix)
            
            # Calculate distances and create dendrogram
            if len(matrix) > 2:
                distances = pdist(matrix, metric='jaccard')
                linkage_matrix = linkage(distances, method='ward')
                
                plt.figure(figsize=(10, 6))
                dendrogram(linkage_matrix, labels=query_labels, leaf_rotation=90)
                plt.title(f'Topic {topic_num} - Query Relationships\n(Based on shared keywords)')
                plt.xlabel('Query')
                plt.ylabel('Distance')
                plt.tight_layout()
                
                # Save
                file_path = self.output_dir / f'topic_{topic_num}_hierarchy.png'
                plt.savefig(file_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved hierarchy: {file_path.name}")
                
        except Exception as e:
            print(f"  Could not create hierarchy: {e}")
    
    def create_summary(self, results):
        """Create summary of findings."""
        print("\n" + "="*60)
        print("SUMMARY OF FINDINGS")
        print("="*60)
        
        summary_lines = []
        summary_lines.append("HIERARCHICAL TOPIC ANALYSIS SUMMARY")
        summary_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        summary_lines.append("")
        
        for result in results:
            if result:
                summary_lines.append(f"\nTOPIC {result['topic_num']}:")
                summary_lines.append(f"  Queries: {result['num_queries']}")
                summary_lines.append(f"  Documents: {result['documents']:,}")
                
                if result['common_keywords']:
                    summary_lines.append("  Common Keywords:")
                    for kw in result['common_keywords'][:5]:
                        summary_lines.append(f"    • {kw['keyword']}: {kw['count']} queries ({kw['percentage']:.0f}%)")
                else:
                    summary_lines.append("  No common keywords")
        
        # Save summary
        summary_file = self.output_dir / f'summary_{self.timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"\nSaved summary to: {summary_file.name}")
        
        # Print key findings
        print("\nKEY FINDINGS:")
        for result in results:
            if result and result['common_keywords']:
                top_kw = result['common_keywords'][0]
                print(f"  Topic {result['topic_num']}: '{top_kw['keyword']}' appears in {top_kw['count']} queries")
    
    def run(self):
        """Run analysis."""
        print("="*60)
        print("STARTING ANALYSIS")
        print("="*60)
        
        # Load data
        df = self.load_data()
        
        # Analyze topics 0-4
        results = []
        for topic_num in range(5):
            result = self.analyze_topic(df, topic_num)
            if result:
                results.append(result)
        
        # Create summary
        self.create_summary(results)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results saved in: {self.output_dir}")
        
        return results

def main():
    """Main function."""
    analyzer = SimpleHierarchicalAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()