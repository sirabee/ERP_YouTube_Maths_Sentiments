#!/usr/bin/env python3
"""
Sort merged_topic_info.csv by Topic column in ascending order
"""

import pandas as pd

def sort_merged_topic_info():
    """Sort the merged topic info CSV by Topic column."""
    
    # File paths
    input_file = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis/BERTopic HDBSCAN Per Query 20250720/merged_topic_info/merged_topic_info.csv"
    output_file = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/src/bertopic_analysis/BERTopic HDBSCAN Per Query 20250720/merged_topic_info/merged_topic_info_sorted.csv"
    
    print("Loading merged_topic_info.csv...")
    
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Original Topic range: {df['Topic'].min()} to {df['Topic'].max()}")
    
    # Sort by Topic column in ascending order
    df_sorted = df.sort_values('Topic', ascending=True)
    
    # Save the sorted file
    df_sorted.to_csv(output_file, index=False)
    
    print(f"✓ Sorted file saved as: {output_file}")
    print(f"✓ After sorting - Topic range: {df_sorted['Topic'].min()} to {df_sorted['Topic'].max()}")
    
    # Show first few rows to verify sorting
    print("\nFirst 10 rows after sorting:")
    print(df_sorted[['Topic', 'Count', 'Name', 'query']].head(10).to_string(index=False))
    
    return output_file

if __name__ == "__main__":
    sort_merged_topic_info()