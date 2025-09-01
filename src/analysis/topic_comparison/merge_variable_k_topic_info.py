#!/usr/bin/env python3
"""
Merge Variable K-means topic_info files into master document
Creates merged_topic_info.csv similar to HDBSCAN structure
"""

import pandas as pd
import os
import glob

def merge_variable_k_topic_info():
    """Merge all Variable K-means topic_info files into one master document."""
    
    # Base path for Variable K-means results
    base_path = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis/optimised_variable_k_phase_4_20250722_224755"
    
    # Output directory (create if doesn't exist)
    output_dir = f"{base_path}/merged_topic_info"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MERGING VARIABLE K-MEANS TOPIC INFO FILES")
    print("=" * 60)
    
    # Find all topic_info files
    pattern = f"{base_path}/*/data/topic_info_*.csv"
    topic_files = glob.glob(pattern)
    
    print(f"Found {len(topic_files)} topic_info files")
    
    merged_data = []
    processed_queries = []
    
    for file_path in topic_files:
        # Extract query name from file path
        query = os.path.basename(file_path).replace('topic_info_', '').replace('.csv', '')
        
        try:
            # Load the topic info file
            df = pd.read_csv(file_path)
            
            # Add query column to identify source
            df['query'] = query
            
            # Append to merged data
            merged_data.append(df)
            processed_queries.append(query)
            
            print(f"✓ Processed {query}: {len(df)} topics")
            
        except Exception as e:
            print(f"⚠ Warning: Could not process {file_path}: {e}")
    
    if not merged_data:
        print("❌ No topic info files could be processed!")
        return None
    
    # Combine all data
    print(f"\nCombining data from {len(merged_data)} files...")
    combined_df = pd.concat(merged_data, ignore_index=True)
    
    # Sort by Topic then by query for consistency
    combined_df = combined_df.sort_values(['Topic', 'query'], ascending=[True, True])
    
    # Save merged file
    merged_file = f"{output_dir}/merged_topic_info.csv"
    combined_df.to_csv(merged_file, index=False)
    
    # Save sorted file
    sorted_file = f"{output_dir}/merged_topic_info_sorted.csv"
    combined_df.to_csv(sorted_file, index=False)
    
    # Generate summary statistics
    total_records = len(combined_df)
    unique_topics = combined_df['Topic'].nunique()
    unique_queries = combined_df['query'].nunique()
    topic_range = f"{combined_df['Topic'].min()} to {combined_df['Topic'].max()}"
    
    print(f"\n" + "=" * 60)
    print("MERGE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total records: {total_records:,}")
    print(f"Unique topics: {unique_topics}")
    print(f"Unique queries: {unique_queries}")
    print(f"Topic range: {topic_range}")
    print(f"\nFiles saved:")
    print(f"├── {merged_file}")
    print(f"└── {sorted_file}")
    
    # Show preview
    print(f"\nFirst 10 rows:")
    preview_cols = ['Topic', 'Count', 'Name', 'query']
    print(combined_df[preview_cols].head(10).to_string(index=False))
    
    return merged_file

if __name__ == "__main__":
    merge_variable_k_topic_info()