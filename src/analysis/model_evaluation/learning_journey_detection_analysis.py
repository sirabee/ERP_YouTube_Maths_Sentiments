#!/usr/bin/env python3
"""
Learning Journey Detection Analysis
Compares how well hierarchical vs original methods detect learning journeys
compared to manual annotations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

def load_learning_journey_data():
    """Load manual annotations with learning journey labels."""
    base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
    
    print("LEARNING JOURNEY DETECTION ANALYSIS")
    print("=" * 50)
    
    # Load consensus annotations (has learning journey labels)
    consensus_file = base_path / "results" / "analysis" / "model_evaluation" / "consensus_annotations_20250813_222930.csv"
    consensus_df = pd.read_csv(consensus_file)
    
    # Load detailed results from enhanced analysis
    enhanced_files = list((base_path / "results" / "analysis" / "enhanced_aggregation").glob("enhanced_aggregation_comparison_*.csv"))
    if not enhanced_files:
        print("Error: Enhanced aggregation results not found")
        return None, None
    
    latest_enhanced = max(enhanced_files, key=lambda x: x.stat().st_mtime)
    enhanced_df = pd.read_csv(latest_enhanced)
    
    print(f"Loaded {len(consensus_df)} samples with manual learning journey annotations")
    print(f"Loaded {len(enhanced_df)} samples with method predictions")
    
    # Check learning journey distribution in manual annotations
    journey_dist = consensus_df['consensus_journey'].value_counts()
    print(f"\nManual learning journey distribution:")
    for label, count in journey_dist.items():
        print(f"  {label}: {count} ({count/len(consensus_df)*100:.1f}%)")
    
    return consensus_df, enhanced_df

def analyze_learning_journey_detection_by_sentiment():
    """Analyze how methods detect learning journeys through sentiment progression."""
    
    consensus_df, enhanced_df = load_learning_journey_data()
    if consensus_df is None:
        return
    
    # Merge datasets on comment_text
    merged_df = consensus_df.merge(enhanced_df, on='comment_text', how='inner')
    
    print(f"\nMerged dataset: {len(merged_df)} samples")
    
    # Identify manually annotated learning journeys
    manual_journeys = merged_df[merged_df['consensus_journey'] == 'yes'].copy()
    manual_non_journeys = merged_df[merged_df['consensus_journey'] == 'no'].copy()
    
    print(f"\nManual learning journeys: {len(manual_journeys)}")
    print(f"Manual non-journeys: {len(manual_non_journeys)}")
    
    if len(manual_journeys) == 0:
        print("No learning journeys found in manual annotations")
        return
    
    print(f"\n1. LEARNING JOURNEY DETECTION THROUGH SENTIMENT PATTERNS")
    print("=" * 60)
    
    # For each method, check how many learning journeys they correctly identify as positive
    methods = ['original', 'hierarchical', 'absa', 'ensemble']
    method_names = ['Original', 'Hierarchical', 'ABSA', 'Ensemble']
    
    print(f"\nA. LEARNING JOURNEY SENTIMENT DETECTION:")
    print("-" * 45)
    print(f"Manual learning journeys should ideally be detected as 'positive'")
    print(f"(since they represent positive educational progression)")
    
    journey_results = {}
    for method, name in zip(methods, method_names):
        pred_col = f"{method}_prediction"
        if pred_col in merged_df.columns:
            # How many learning journeys does this method predict as positive?
            journey_positive = manual_journeys[manual_journeys[pred_col] == 'positive']
            journey_negative = manual_journeys[manual_journeys[pred_col] == 'negative']
            journey_neutral = manual_journeys[manual_journeys[pred_col] == 'neutral']
            
            positive_rate = len(journey_positive) / len(manual_journeys) * 100
            negative_rate = len(journey_negative) / len(manual_journeys) * 100
            neutral_rate = len(journey_neutral) / len(manual_journeys) * 100
            
            journey_results[name] = {
                'positive_rate': positive_rate,
                'negative_rate': negative_rate,
                'neutral_rate': neutral_rate,
                'total_journeys': len(manual_journeys)
            }
            
            print(f"\n{name}:")
            print(f"  Learning journeys → Positive: {len(journey_positive)}/{len(manual_journeys)} ({positive_rate:.1f}%)")
            print(f"  Learning journeys → Negative: {len(journey_negative)}/{len(manual_journeys)} ({negative_rate:.1f}%)")
            print(f"  Learning journeys → Neutral:  {len(journey_neutral)}/{len(manual_journeys)} ({neutral_rate:.1f}%)")
    
    print(f"\nB. NON-JOURNEY SENTIMENT DISTRIBUTION:")
    print("-" * 40)
    print(f"Non-learning journeys should have varied sentiment distribution")
    
    for method, name in zip(methods, method_names):
        pred_col = f"{method}_prediction"
        if pred_col in merged_df.columns:
            non_journey_dist = manual_non_journeys[pred_col].value_counts()
            print(f"\n{name} on non-journeys:")
            for sentiment, count in non_journey_dist.items():
                print(f"  {sentiment}: {count} ({count/len(manual_non_journeys)*100:.1f}%)")
    
    return merged_df, manual_journeys, manual_non_journeys, journey_results

def detect_sentiment_progression_patterns(merged_df, manual_journeys):
    """Analyze which method best detects actual sentiment progression patterns."""
    
    print(f"\n2. SENTIMENT PROGRESSION PATTERN ANALYSIS")
    print("=" * 50)
    
    # Load the original enhanced analysis to get metadata about progression detection
    base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
    
    print(f"Analyzing progression patterns in {len(manual_journeys)} manual learning journeys...")
    
    # For each learning journey, show actual sentiment progression
    print(f"\nDETAILED LEARNING JOURNEY EXAMPLES:")
    print("-" * 40)
    
    for i, (idx, row) in enumerate(manual_journeys.head(5).iterrows()):
        comment_text = row['comment_text']
        manual_sentiment = row['consensus_sentiment']
        
        print(f"\nLearning Journey {i+1}:")
        print(f"Text: \"{comment_text[:100]}...\"")
        print(f"Manual consensus sentiment: {manual_sentiment}")
        print(f"Method predictions:")
        
        methods = ['original', 'hierarchical', 'absa', 'ensemble']
        method_names = ['Original', 'Hierarchical', 'ABSA', 'Ensemble']
        
        for method, name in zip(methods, method_names):
            pred_col = f"{method}_prediction"
            if pred_col in merged_df.columns:
                prediction = row[pred_col]
                match = '✓' if prediction == manual_sentiment else '✗'
                print(f"  {name:<12}: {prediction} {match}")

def analyze_method_capability_for_progression():
    """Analyze which method is theoretically better for detecting progression."""
    
    print(f"\n3. METHODOLOGICAL CAPABILITY ANALYSIS")
    print("=" * 45)
    
    print(f"""
THEORETICAL LEARNING JOURNEY DETECTION CAPABILITY:

ORIGINAL METHOD:
• Sentence-level analysis: ✓ Can detect sentence progression
• Learning journey detection: ✓ Basic negative→positive detection
• Progression weighting: ✓ Simple majority vote
• Educational context: ✗ No domain-specific processing

HIERARCHICAL METHOD:
• Phrase-level analysis: ✓✓ Can detect sub-sentence progression
• Position weighting: ✓✓ Later sentences weighted more heavily
• Multiple granularity: ✓✓ Phrases → Sentences → Comment
• Educational context: ✗ No domain-specific processing

ABSA METHOD:
• Educational aspects: ✓✓ Detects gratitude, understanding, difficulty
• Domain expertise: ✓✓ Specifically designed for educational content
• Learning indicators: ✓✓ Identifies 'learning_progress' keywords
• Progression detection: ✗ Less focused on temporal progression

ENSEMBLE METHOD:
• Combined approach: ✓ Leverages all methods
• Balanced decision: ✓ Reduces individual method bias
• Educational weighting: ✓ Increases ABSA weight when educational aspects detected
• Consensus validation: ✓ Higher confidence when methods agree
""")

def calculate_learning_journey_metrics(merged_df, manual_journeys, manual_non_journeys):
    """Calculate specific metrics for learning journey detection."""
    
    print(f"\n4. LEARNING JOURNEY DETECTION METRICS")
    print("=" * 45)
    
    methods = ['original', 'hierarchical', 'absa', 'ensemble']
    method_names = ['Original', 'Hierarchical', 'ABSA', 'Ensemble']
    
    print(f"\nMETRIC 1: Learning Journey → Positive Sentiment Accuracy")
    print(f"(How well each method identifies learning journeys as positive)")
    print("-" * 60)
    
    best_method = None
    best_score = 0
    
    for method, name in zip(methods, method_names):
        pred_col = f"{method}_prediction"
        if pred_col in merged_df.columns:
            # Learning journeys correctly identified as positive
            journey_positive = manual_journeys[manual_journeys[pred_col] == 'positive']
            accuracy = len(journey_positive) / len(manual_journeys)
            
            print(f"{name:<15}: {len(journey_positive):2d}/{len(manual_journeys)} ({accuracy*100:5.1f}%)")
            
            if accuracy > best_score:
                best_score = accuracy
                best_method = name
    
    print(f"\nBest method for learning journey detection: {best_method} ({best_score*100:.1f}%)")
    
    print(f"\nMETRIC 2: Overall Sentiment Accuracy on Learning Journeys")
    print(f"(Agreement with manual consensus sentiment on learning journeys)")
    print("-" * 60)
    
    best_overall = None
    best_overall_score = 0
    
    for method, name in zip(methods, method_names):
        pred_col = f"{method}_prediction"
        if pred_col in merged_df.columns:
            # Overall accuracy on learning journey comments
            correct = manual_journeys[manual_journeys[pred_col] == manual_journeys['consensus_sentiment']]
            accuracy = len(correct) / len(manual_journeys)
            
            print(f"{name:<15}: {len(correct):2d}/{len(manual_journeys)} ({accuracy*100:5.1f}%)")
            
            if accuracy > best_overall_score:
                best_overall_score = accuracy
                best_overall = name
    
    print(f"\nBest overall accuracy on learning journeys: {best_overall} ({best_overall_score*100:.1f}%)")
    
    return best_method, best_overall

def main():
    """Main analysis function."""
    
    # Analyze learning journey detection through sentiment
    merged_df, manual_journeys, manual_non_journeys, journey_results = analyze_learning_journey_detection_by_sentiment()
    if merged_df is None:
        return
    
    # Analyze progression patterns
    detect_sentiment_progression_patterns(merged_df, manual_journeys)
    
    # Analyze methodological capabilities
    analyze_method_capability_for_progression()
    
    # Calculate specific metrics
    best_method, best_overall = calculate_learning_journey_metrics(merged_df, manual_journeys, manual_non_journeys)
    
    print(f"\n" + "=" * 60)
    print(f"CONCLUSION: LEARNING JOURNEY DETECTION")
    print(f"=" * 60)
    print(f"""
KEY FINDINGS:

1. Best method for detecting learning journeys as positive: {best_method}
2. Best overall accuracy on learning journey comments: {best_overall}

METHODOLOGICAL INSIGHTS:
• Hierarchical method should theoretically be best due to progression weighting
• ABSA method has domain expertise but may miss temporal progression
• Original method provides baseline sentence-level progression detection
• Ensemble balances different strengths but may dilute progression signals

RECOMMENDATION:
For learning journey detection in educational sentiment analysis,
consider the hierarchical approach with enhanced educational domain features.
""")

if __name__ == "__main__":
    main()