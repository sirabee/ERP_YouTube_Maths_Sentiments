#!/usr/bin/env python3
"""
Fixed Hybrid Sampling Script for 200-Comment Validation
Generates sample datasets and annotation templates
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime

def load_datasets():
    """Load both HDBSCAN and Variable K-means datasets"""
    
    # Correct file paths
    hdbscan_path = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/Sentiment_Analysis/HDBSCAN/sentence_sentiment_analysis_20250726_221258/comments_with_sentence_sentiment.csv"
    vark_path = "/Users/siradbihi/Desktop/MScDataScience/YouTube_Mathematics_Sentiment_Analysis/Sentiment_Analysis/Variable_K_Means/variable_k_sentence_sentiment_analysis_20250731_010046/variable_k_comments_with_sentence_sentiment.csv"
    
    try:
        hdbscan_df = pd.read_csv(hdbscan_path)
        vark_df = pd.read_csv(vark_path)
        
        print(f"Loaded HDBSCAN dataset: {len(hdbscan_df):,} comments")
        print(f"Loaded Variable K-means dataset: {len(vark_df):,} comments")
        
        # Validate required columns
        required_cols = ['comment_text', 'sentence_progression', 'final_sentiment_weight']
        for dataset_name, df in [("HDBSCAN", hdbscan_df), ("Variable K-means", vark_df)]:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"Warning: {dataset_name} missing columns: {missing}")
        
        return hdbscan_df, vark_df
        
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return None, None

def create_stratified_sample(df, dataset_name, total_size=200):
    """Create stratified sample with clear methodology"""
    
    print(f"\nCreating sample for {dataset_name} (target: {total_size} comments)")
    
    # Check required columns
    sentiment_col = 'sentence_level_sentiment' if 'sentence_level_sentiment' in df.columns else 'overall_sentiment'
    
    samples = []
    used_indices = set()
    
    # 1. Random sample (40%)
    random_n = int(total_size * 0.4)
    random_sample = df.sample(n=min(random_n, len(df)), random_state=42)
    samples.append(random_sample)
    used_indices.update(random_sample.index)
    print(f"  ✓ Random: {len(random_sample)} comments")
    
    # 2. Learning journeys (30%)
    remaining_df = df[~df.index.isin(used_indices)]
    lj_n = int(total_size * 0.3)
    
    if 'sentiment_progression' in df.columns:
        learning_journeys = remaining_df[remaining_df['sentiment_progression'] == 'learning_journey']
        lj_sample_n = min(lj_n, len(learning_journeys))
        if lj_sample_n > 0:
            lj_sample = learning_journeys.nlargest(lj_sample_n, 'final_sentiment_weight')
            samples.append(lj_sample)
            used_indices.update(lj_sample.index)
            print(f"  ✓ Learning journeys: {len(lj_sample)} comments")
    
    # 3. High confidence (20%)
    remaining_df = df[~df.index.isin(used_indices)]
    hc_n = int(total_size * 0.2)
    
    high_conf = remaining_df[remaining_df['final_sentiment_weight'] >= 0.85]
    hc_sample_n = min(hc_n, len(high_conf))
    if hc_sample_n > 0:
        hc_sample = high_conf.nlargest(hc_sample_n, 'final_sentiment_weight')
        samples.append(hc_sample)
        used_indices.update(hc_sample.index)
        print(f"  ✓ High confidence: {len(hc_sample)} comments")
    
    # 4. Fill remaining with random
    current_total = sum(len(s) for s in samples)
    if current_total < total_size:
        remaining_df = df[~df.index.isin(used_indices)]
        fill_n = total_size - current_total
        
        if len(remaining_df) > 0:
            fill_sample = remaining_df.sample(n=min(fill_n, len(remaining_df)), random_state=42)
            samples.append(fill_sample)
            print(f"  ✓ Fill remaining: {len(fill_sample)} comments")
    
    # Combine
    final_sample = pd.concat(samples, ignore_index=True)
    print(f"  → Final sample: {len(final_sample)} comments")
    
    return final_sample

def create_annotation_template(sample_df, dataset_name):
    """Create blinded annotation template matching model outputs"""
    
    # Create blinded template - only comment text and annotation fields that match model outputs
    template = pd.DataFrame({
        'comment_id': range(1, len(sample_df) + 1),
        'comment_text': sample_df['comment_text'].values,
        'sentence_count': sample_df.get('sentence_count', 1),
        
        # Manual annotation columns (matching model outputs)
        'manual_sentiment': '',  # positive/negative/neutral
        'manual_learning_journey': '',  # yes/no (matches sentence_progression == 'learning_journey')
        'manual_has_transition': '',  # yes/no (matches has_transition)
        'manual_confidence_0_to_1': '',  # 0.0-1.0 scale (matches final_sentiment_weight)
        
        # Quality control
        'annotator_notes': '',
        'annotation_difficulty': ''  # easy/medium/hard
    })
    
    return template

def create_evaluation_template(sample_df, dataset_name):
    """Create evaluation template with model predictions for comparison"""
    
    # Handle different column names
    sentiment_col = 'sentence_level_sentiment' if 'sentence_level_sentiment' in sample_df.columns else 'overall_sentiment'
    
    # Handle learning journey column safely
    if 'sentiment_progression' in sample_df.columns:
        learning_journey_values = sample_df['sentiment_progression'].apply(lambda x: x == 'learning_journey')
    else:
        learning_journey_values = [False] * len(sample_df)
    
    template = pd.DataFrame({
        'comment_id': range(1, len(sample_df) + 1),
        'dataset_type': dataset_name,
        'comment_text': sample_df['comment_text'].values,
        'sentence_count': sample_df.get('sentence_count', 1),
        
        # System predictions (for evaluation)
        'system_sentiment': sample_df[sentiment_col].values,
        'system_confidence': sample_df['final_sentiment_weight'].values,
        'system_learning_journey': learning_journey_values,
        
        # Manual annotation results (to be filled from blinded template)
        'manual_sentiment': '',
        'manual_learning_journey': '',
        'manual_has_transition': '',
        'manual_confidence_0_to_1': '',
        'annotator_notes': '',
        'annotation_difficulty': '',
        
        # Evaluation metrics (calculated later)
        'sentiment_agreement': '',
        'learning_journey_agreement': '',
        'transition_agreement': '',
        'confidence_difference': ''
    })
    
    return template

def main():
    """Generate validation samples"""
    
    print("=" * 60)
    print("SENTIMENT ANALYSIS VALIDATION SAMPLE GENERATION")
    print("=" * 60)
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    
    # Load data
    hdbscan_df, vark_df = load_datasets()
    if hdbscan_df is None or vark_df is None:
        return
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"validation_samples_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Generate samples
    hdbscan_sample = create_stratified_sample(hdbscan_df, "HDBSCAN", 200)
    vark_sample = create_stratified_sample(vark_df, "Variable_K", 200)
    
    # Create blinded annotation templates (for annotators)
    hdbscan_template = create_annotation_template(hdbscan_sample, "HDBSCAN")
    vark_template = create_annotation_template(vark_sample, "Variable_K")
    
    # Create evaluation templates (with model predictions for analysis)
    hdbscan_eval = create_evaluation_template(hdbscan_sample, "HDBSCAN")
    vark_eval = create_evaluation_template(vark_sample, "Variable_K")
    
    # Save blinded files (for annotators)
    hdbscan_template.to_csv(output_dir / "HDBSCAN_annotation_BLINDED.csv", index=False)
    vark_template.to_csv(output_dir / "Variable_K_annotation_BLINDED.csv", index=False)
    
    # Save evaluation files (for researchers)
    hdbscan_eval.to_csv(output_dir / "HDBSCAN_evaluation_template.csv", index=False)
    vark_eval.to_csv(output_dir / "Variable_K_evaluation_template.csv", index=False)
    
    print(f"\n" + "=" * 60)
    print("SAMPLE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Files saved to: {output_dir}")
    print(f"✓ HDBSCAN sample: {len(hdbscan_template)} comments")
    print(f"✓ Variable K-means sample: {len(vark_template)} comments")
    print(f"✓ Total validation samples: {len(hdbscan_template) + len(vark_template)}")
    print(f"✓ Per dataset: 200 comments each")
    print(f"✓ Sample composition per dataset:")
    print(f"  - 80 random comments (40%)")
    print(f"  - 60 learning journeys (30%)")
    print(f"  - 40 high-confidence (20%)")
    print(f"  - 20 remaining/edge cases (10%)")
    print(f"\n📁 Files generated:")
    print(f"├── HDBSCAN_annotation_BLINDED.csv (for annotators)")
    print(f"├── Variable_K_annotation_BLINDED.csv (for annotators)")
    print(f"├── HDBSCAN_evaluation_template.csv (for researchers)")
    print(f"└── Variable_K_evaluation_template.csv (for researchers)")
    print(f"\n⚠️  IMPORTANT: Use BLINDED files for annotation to avoid bias!")

if __name__ == "__main__":
    main()