#!/usr/bin/env python3
"""
Topic 1 Keyword Comparison: HDBSCAN vs Variable K-means
Creates a clean table comparing keywords for Topic 1 across queries
"""

import pandas as pd
import os
import glob
import ast

def get_topic_keywords(base_path, query, topic_num):
    """Extract keywords for specified topic number."""
    if "optimised_variable_k" in base_path:
        file_path = f"{base_path}/{query}/data/topic_info_{query}.csv"
    else:
        file_path = f"{base_path}/{query}/topic_info_{query}.csv"
    
    if not os.path.exists(file_path):
        return []
    
    try:
        df = pd.read_csv(file_path)
        topic_row = df[df['Topic'] == topic_num]
        if topic_row.empty:
            return []
        
        keywords_str = topic_row.iloc[0]['Representation']
        keywords = ast.literal_eval(keywords_str)
        return keywords[:5]  # Top 5 keywords
    except:
        return []

def main():
    # Paths
    hdbscan_path = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis/BERTopic HDBSCAN Per Query 20250720/bertopic_complete_pipeline_analysis_20250720_230249"
    variable_k_path = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis/optimised_variable_k_phase_4_20250722_224755"
    
    # Get common queries
    hdbscan_queries = set()
    for file_path in glob.glob(f"{hdbscan_path}/*/topic_info_*.csv"):
        query = os.path.basename(file_path).replace('topic_info_', '').replace('.csv', '')
        hdbscan_queries.add(query)
    
    variable_k_queries = set()
    for file_path in glob.glob(f"{variable_k_path}/*/data/topic_info_*.csv"):
        query = os.path.basename(file_path).replace('topic_info_', '').replace('.csv', '')
        variable_k_queries.add(query)
    
    common_queries = sorted(hdbscan_queries.intersection(variable_k_queries))
    
    # Build comparison table for Topic 1
    results = []
    for query in common_queries:
        hdbscan_keywords = get_topic_keywords(hdbscan_path, query, 1)
        variable_k_keywords = get_topic_keywords(variable_k_path, query, 1)
        
        results.append({
            'Query': query.replace('_', ' ').title(),
            'HDBSCAN': ', '.join(hdbscan_keywords) if hdbscan_keywords else 'No data',
            'Variable_K': ', '.join(variable_k_keywords) if variable_k_keywords else 'No data'
        })
    
    # Create and save table
    df = pd.DataFrame(results)
    df.to_csv('topic1_keyword_comparison.csv', index=False)
    
    print(f"✓ Analyzed {len(common_queries)} queries for Topic 1")
    print(f"✓ Table saved as: topic1_keyword_comparison.csv")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()