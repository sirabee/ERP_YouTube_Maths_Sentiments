#!/usr/bin/env python3
"""
Calculate model agreement metrics for YouTube BERT and YouTube XLM-RoBERTa.
Output format similar to mapping_summary_20250813_222930.json.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def calculate_agreement_metrics():
    """Calculate agreement metrics similar to mapping_summary format."""
    
    # Load data
    base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
    predictions_file = base_path / "results" / "analysis" / "model_evaluation" / "detailed_model_predictions_20250813_233026.csv"
    
    print(f"Loading predictions from: {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    print(f"Total samples: {len(df)}")
    
    # Calculate metrics for each model
    models = {
        'twitter_roberta': 'pred_twitter_roberta_baseline',
        'youtube_bert': 'pred_youtube_bert_general',
        'youtube_xlm_roberta': 'pred_youtube_xlm-roberta'
    }
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, pred_column in models.items():
        print(f"\nCalculating metrics for {model_name}...")
        
        # Sentiment agreement
        sentiment_matches = (df['consensus_sentiment'] == df[pred_column]).sum()
        sentiment_match_rate = sentiment_matches / len(df)
        
        # Sentiment distributions (convert to regular int for JSON serialization)
        consensus_dist = {k: int(v) for k, v in df['consensus_sentiment'].value_counts().to_dict().items()}
        model_dist = {k: int(v) for k, v in df[pred_column].value_counts().to_dict().items()}
        
        # Create summary in same format as mapping_summary
        model_summary = {
            "total_samples": len(df),
            "timestamp": timestamp,
            "model_name": model_name,
            "model_alignment": {
                "sentiment_match_rate": round(sentiment_match_rate, 3),
                "sentiment_matches": int(sentiment_matches)
            },
            "sentiment_distribution": {
                "consensus": consensus_dist,
                "model": model_dist
            }
        }
        
        results[model_name] = model_summary
        
        # Print summary
        print(f"  Sentiment match rate: {sentiment_match_rate:.3f} ({sentiment_matches}/{len(df)})")
        print(f"  Consensus distribution: {consensus_dist}")
        print(f"  Model distribution: {model_dist}")
    
    # Save results
    output_dir = base_path / "results" / "analysis" / "model_evaluation"
    
    for model_name, summary in results.items():
        output_file = output_dir / f"{model_name}_agreement_metrics_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved {model_name} metrics to: {output_file}")
    
    # Create comparison summary
    comparison = {
        "total_samples": len(df),
        "timestamp": timestamp,
        "model_comparison": {
            "twitter_roberta_agreement": results['twitter_roberta']['model_alignment']['sentiment_match_rate'],
            "youtube_bert_agreement": results['youtube_bert']['model_alignment']['sentiment_match_rate'],
            "youtube_xlm_roberta_agreement": results['youtube_xlm_roberta']['model_alignment']['sentiment_match_rate'],
            "best_model": max(results.keys(), key=lambda k: results[k]['model_alignment']['sentiment_match_rate'])
        },
        "consensus_distribution": results['youtube_bert']['sentiment_distribution']['consensus']
    }
    
    comparison_file = output_dir / f"model_agreement_comparison_{timestamp}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nSaved comparison to: {comparison_file}")
    
    return results, comparison

if __name__ == "__main__":
    results, comparison = calculate_agreement_metrics()
    
    print("\n" + "="*60)
    print("MODEL AGREEMENT SUMMARY")
    print("="*60)
    print(f"Twitter RoBERTa agreement: {results['twitter_roberta']['model_alignment']['sentiment_match_rate']:.1%}")
    print(f"YouTube BERT agreement: {results['youtube_bert']['model_alignment']['sentiment_match_rate']:.1%}")
    print(f"YouTube XLM-RoBERTa agreement: {results['youtube_xlm_roberta']['model_alignment']['sentiment_match_rate']:.1%}")
    print(f"Best performing model: {comparison['model_comparison']['best_model']}")
    
    # Compare to original Twitter RoBERTa performance
    print("\n" + "="*60)
    print("COMPARISON TO ORIGINAL TWITTER ROBERTA PERFORMANCE")
    print("="*60)
    original_twitter_rate = 0.37  # From mapping_summary_20250813_222930.json
    current_twitter_rate = results['twitter_roberta']['model_alignment']['sentiment_match_rate']
    print(f"Original Twitter RoBERTa (mapping_summary): {original_twitter_rate:.1%}")
    print(f"Current Twitter RoBERTa (detailed_predictions): {current_twitter_rate:.1%}")
    print(f"Performance {'matches' if abs(original_twitter_rate - current_twitter_rate) < 0.01 else 'differs by'}: {abs(original_twitter_rate - current_twitter_rate):.1%}")