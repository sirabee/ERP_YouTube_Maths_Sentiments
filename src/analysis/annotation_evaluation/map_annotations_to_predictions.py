"""
Map manual annotations to model predictions for performance evaluation.
Creates consensus ground truth from two annotators and aligns with model outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def load_manual_annotations(annotations_path):
    """
    Load manual annotations from both annotators.
    
    Args:
        annotations_path: Path to annotations directory
        
    Returns:
        tuple: DataFrames for annotator 1 and annotator 2
    """
    ann1_path = annotations_path / "Variable_K_annotation_Annotator1_KK.csv"
    ann2_path = annotations_path / "Variable_K_annotation_Annotator2_EB.csv"
    
    df1 = pd.read_csv(ann1_path)
    df2 = pd.read_csv(ann2_path)
    
    # Strip whitespace from column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    return df1, df2

def standardize_sentiment(value):
    """
    Standardize sentiment labels to consistent format.
    
    Args:
        value: Raw sentiment value
        
    Returns:
        str: Standardized sentiment (positive, negative, neutral)
    """
    if pd.isna(value):
        return 'neutral'
    
    value = str(value).lower().strip()
    
    # Map to standard labels
    if value in ['positive', 'pos']:
        return 'positive'
    elif value in ['negative', 'neg']:
        return 'negative'
    else:
        return 'neutral'

def standardize_binary(value):
    """
    Standardize binary labels (yes/no).
    
    Args:
        value: Raw binary value
        
    Returns:
        str: Standardized binary (yes/no)
    """
    if pd.isna(value):
        return 'no'
    
    value = str(value).lower().strip()
    
    if value in ['yes', 'true', '1', 'y']:
        return 'yes'
    else:
        return 'no'

def create_consensus_annotations(df1, df2):
    """
    Create consensus ground truth from two annotators.
    Uses simple agreement where annotators agree, flags disagreements.
    
    Args:
        df1: Annotator 1 DataFrame
        df2: Annotator 2 DataFrame
        
    Returns:
        DataFrame: Consensus annotations with disagreement flags
    """
    # Standardize annotations
    df1['sentiment_std'] = df1['manual sentiment'].apply(standardize_sentiment)
    df2['sentiment_std'] = df2['manual sentiment'].apply(standardize_sentiment)
    
    df1['journey_std'] = df1['manual learning journey'].apply(standardize_binary)
    df2['journey_std'] = df2['manual learning journey'].apply(standardize_binary)
    
    df1['transition_std'] = df1['manual has transition'].apply(standardize_binary)
    df2['transition_std'] = df2['manual has transition'].apply(standardize_binary)
    
    # Create consensus dataframe
    consensus = pd.DataFrame({
        'comment_id': df1['comment_id'],
        'comment_text': df1['comment_text'],
        
        # Annotator 1
        'ann1_sentiment': df1['sentiment_std'],
        'ann1_journey': df1['journey_std'],
        'ann1_transition': df1['transition_std'],
        'ann1_confidence': df1['manual confidence 0 to 1'],
        
        # Annotator 2
        'ann2_sentiment': df2['sentiment_std'],
        'ann2_journey': df2['journey_std'],
        'ann2_transition': df2['transition_std'],
        'ann2_confidence': df2['manual confidence 0 to 1'],
    })
    
    # Create consensus labels (use agreement, or higher confidence annotator)
    consensus['sentiment_agrees'] = consensus['ann1_sentiment'] == consensus['ann2_sentiment']
    consensus['journey_agrees'] = consensus['ann1_journey'] == consensus['ann2_journey']
    consensus['transition_agrees'] = consensus['ann1_transition'] == consensus['ann2_transition']
    
    # For disagreements, use annotator with higher confidence
    def resolve_disagreement(row, field):
        """Resolve disagreement using confidence scores."""
        if row[f'{field}_agrees']:
            return row[f'ann1_{field}']
        else:
            if row['ann1_confidence'] >= row['ann2_confidence']:
                return row[f'ann1_{field}']
            else:
                return row[f'ann2_{field}']
    
    consensus['consensus_sentiment'] = consensus.apply(
        lambda x: resolve_disagreement(x, 'sentiment'), axis=1
    )
    consensus['consensus_journey'] = consensus.apply(
        lambda x: resolve_disagreement(x, 'journey'), axis=1
    )
    consensus['consensus_transition'] = consensus.apply(
        lambda x: resolve_disagreement(x, 'transition'), axis=1
    )
    
    # Add consensus confidence (average when agree, single when disagree)
    consensus['consensus_confidence'] = consensus.apply(
        lambda x: (x['ann1_confidence'] + x['ann2_confidence']) / 2 
        if x['sentiment_agrees'] 
        else max(x['ann1_confidence'], x['ann2_confidence']), 
        axis=1
    )
    
    return consensus

def load_model_predictions(predictions_path):
    """
    Load model predictions from Variable K sentiment analysis.
    
    Args:
        predictions_path: Path to model predictions CSV
        
    Returns:
        DataFrame: Model predictions with sentiment and features
    """
    # Load sentence-level predictions
    df = pd.read_csv(predictions_path)
    
    # Group by comment_id to get comment-level predictions
    comment_predictions = df.groupby('comment_id').agg({
        'original_text': 'first',
        'overall_sentiment': 'first',
        'has_transition': 'first',
        'avg_confidence': 'first',
        'sentence_count': 'first'
    }).reset_index()
    
    # Standardize model sentiment labels
    comment_predictions['model_sentiment'] = comment_predictions['overall_sentiment'].apply(
        standardize_sentiment
    )
    
    # Model doesn't predict learning journey directly, but we can infer from text patterns
    # For now, we'll mark this as not available
    comment_predictions['model_journey'] = 'not_predicted'
    
    # Standardize transition predictions
    comment_predictions['model_transition'] = comment_predictions['has_transition'].apply(
        lambda x: 'yes' if x else 'no'
    )
    
    return comment_predictions

def align_annotations_with_predictions(consensus, predictions):
    """
    Align manual annotations with model predictions by comment_id.
    
    Args:
        consensus: Consensus annotations DataFrame
        predictions: Model predictions DataFrame
        
    Returns:
        DataFrame: Aligned dataset with both annotations and predictions
    """
    # Merge on comment_id
    aligned = pd.merge(
        consensus,
        predictions[['comment_id', 'model_sentiment', 'model_journey', 
                    'model_transition', 'avg_confidence']],
        on='comment_id',
        how='inner'
    )
    
    # Rename model confidence column
    aligned = aligned.rename(columns={'avg_confidence': 'model_confidence'})
    
    # Add comparison flags
    aligned['sentiment_match'] = (
        aligned['consensus_sentiment'] == aligned['model_sentiment']
    )
    aligned['transition_match'] = (
        aligned['consensus_transition'] == aligned['model_transition']
    )
    
    return aligned

def generate_mapping_summary(aligned):
    """
    Generate summary statistics for the mapping.
    
    Args:
        aligned: Aligned DataFrame
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_samples': len(aligned),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        
        'annotation_consensus': {
            'sentiment_agreement_rate': aligned['sentiment_agrees'].mean(),
            'journey_agreement_rate': aligned['journey_agrees'].mean(),
            'transition_agreement_rate': aligned['transition_agrees'].mean(),
            'avg_consensus_confidence': aligned['consensus_confidence'].mean()
        },
        
        'model_alignment': {
            'sentiment_match_rate': aligned['sentiment_match'].mean(),
            'transition_match_rate': aligned['transition_match'].mean(),
            'avg_model_confidence': aligned['model_confidence'].mean()
        },
        
        'sentiment_distribution': {
            'consensus': aligned['consensus_sentiment'].value_counts().to_dict(),
            'model': aligned['model_sentiment'].value_counts().to_dict()
        },
        
        'transition_distribution': {
            'consensus': aligned['consensus_transition'].value_counts().to_dict(),
            'model': aligned['model_transition'].value_counts().to_dict()
        }
    }
    
    return summary

def main():
    """
    Main function to map annotations to predictions.
    """
    # Set up paths
    base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
    annotations_path = base_path / "data" / "annotations" / "samples"
    predictions_path = base_path / "results" / "models" / "sentiment_analysis" / \
        "variable_k_sentence_sentiment_analysis_20250731_010046" / \
        "variable_k_sentence_level_details.csv"
    output_path = base_path / "results" / "analysis" / "model_evaluation"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Mapping Manual Annotations to Model Predictions")
    print("=" * 60)
    
    # Load manual annotations
    print("\n1. Loading manual annotations...")
    df1, df2 = load_manual_annotations(annotations_path)
    print(f"   Loaded {len(df1)} annotations from each annotator")
    
    # Create consensus
    print("\n2. Creating consensus annotations...")
    consensus = create_consensus_annotations(df1, df2)
    
    # Calculate agreement stats
    sentiment_agrees = consensus['sentiment_agrees'].sum()
    journey_agrees = consensus['journey_agrees'].sum()
    transition_agrees = consensus['transition_agrees'].sum()
    
    print(f"   Sentiment agreement: {sentiment_agrees}/{len(consensus)} samples")
    print(f"   Journey agreement: {journey_agrees}/{len(consensus)} samples")
    print(f"   Transition agreement: {transition_agrees}/{len(consensus)} samples")
    
    # Load model predictions
    print("\n3. Loading model predictions...")
    predictions = load_model_predictions(predictions_path)
    print(f"   Loaded predictions for {len(predictions)} comments")
    
    # Align annotations with predictions
    print("\n4. Aligning annotations with predictions...")
    aligned = align_annotations_with_predictions(consensus, predictions)
    print(f"   Successfully aligned {len(aligned)} samples")
    
    if len(aligned) < len(consensus):
        missing = len(consensus) - len(aligned)
        print(f"   WARNING: {missing} annotations not found in model predictions")
        
        # Find missing comment IDs
        missing_ids = set(consensus['comment_id']) - set(aligned['comment_id'])
        if missing_ids:
            print(f"   Missing comment IDs: {sorted(missing_ids)}")
    
    # Generate summary
    print("\n5. Generating mapping summary...")
    summary = generate_mapping_summary(aligned)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save aligned dataset
    aligned_file = output_path / f"aligned_annotations_predictions_{timestamp}.csv"
    aligned.to_csv(aligned_file, index=False)
    print(f"\n   Saved aligned dataset: {aligned_file.name}")
    
    # Save consensus annotations
    consensus_file = output_path / f"consensus_annotations_{timestamp}.csv"
    consensus.to_csv(consensus_file, index=False)
    print(f"   Saved consensus annotations: {consensus_file.name}")
    
    # Save summary
    summary_file = output_path / f"mapping_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved mapping summary: {summary_file.name}")
    
    # Print summary report
    print("\n" + "=" * 60)
    print("MAPPING SUMMARY")
    print("=" * 60)
    print(f"Total samples aligned: {summary['total_samples']}")
    print(f"\nAnnotator Agreement Rates:")
    print(f"  Sentiment: {summary['annotation_consensus']['sentiment_agreement_rate']:.1%}")
    print(f"  Journey: {summary['annotation_consensus']['journey_agreement_rate']:.1%}")
    print(f"  Transition: {summary['annotation_consensus']['transition_agreement_rate']:.1%}")
    print(f"\nModel-Human Alignment:")
    print(f"  Sentiment match: {summary['model_alignment']['sentiment_match_rate']:.1%}")
    print(f"  Transition match: {summary['model_alignment']['transition_match_rate']:.1%}")
    print(f"\nConsensus Sentiment Distribution:")
    for sentiment, count in summary['sentiment_distribution']['consensus'].items():
        print(f"  {sentiment}: {count}")
    print(f"\nModel Sentiment Distribution:")
    for sentiment, count in summary['sentiment_distribution']['model'].items():
        print(f"  {sentiment}: {count}")
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()