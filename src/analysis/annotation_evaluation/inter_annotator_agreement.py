"""
Calculate inter-annotator agreement (Cohen's Kappa) for manual sentiment annotations.
Assesses agreement between two annotators on sentiment, learning journey, and transitions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from pathlib import Path
import json
from datetime import datetime

def load_annotations(base_path):
    """
    Load annotation data from both annotators.
    
    Args:
        base_path: Path to annotations directory
        
    Returns:
        tuple: DataFrames for annotator 1 and annotator 2
    """
    annotator1_path = base_path / "Variable_K_annotation_Annotator1_KK.csv"
    annotator2_path = base_path / "Variable_K_annotation_Annotator2_EB.csv"
    
    # Load CSV files
    df1 = pd.read_csv(annotator1_path)
    df2 = pd.read_csv(annotator2_path)
    
    # Ensure consistent column naming
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    return df1, df2

def standardize_labels(series):
    """
    Standardize annotation labels for consistency.
    Handles case differences and common variations.
    
    Args:
        series: Pandas series with annotation labels
        
    Returns:
        series: Standardized labels
    """
    # Convert to lowercase and strip whitespace
    series = series.astype(str).str.lower().str.strip()
    
    # Map common variations to standard labels
    mapping = {
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral',
        'yes': 'yes',
        'no': 'no',
        'true': 'yes',
        'false': 'no'
    }
    
    return series.map(lambda x: mapping.get(x, x))

def calculate_kappa(labels1, labels2, label_name):
    """
    Calculate Cohen's Kappa for a specific annotation type.
    
    Args:
        labels1: Labels from annotator 1
        labels2: Labels from annotator 2
        label_name: Name of the annotation type (for reporting)
        
    Returns:
        dict: Results including kappa score and interpretation
    """
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(labels1, labels2)
    
    # Get unique labels for confusion matrix
    unique_labels = sorted(set(labels1) | set(labels2))
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels1, labels2, labels=unique_labels)
    
    # Calculate agreement percentage
    agreement_count = sum(l1 == l2 for l1, l2 in zip(labels1, labels2))
    total_count = len(labels1)
    agreement_pct = (agreement_count / total_count) * 100
    
    # Interpret Kappa score (Landis & Koch, 1977)
    if kappa < 0:
        interpretation = "Poor (less than chance)"
    elif kappa <= 0.20:
        interpretation = "Slight"
    elif kappa <= 0.40:
        interpretation = "Fair"
    elif kappa <= 0.60:
        interpretation = "Moderate"
    elif kappa <= 0.80:
        interpretation = "Substantial"
    else:
        interpretation = "Almost perfect"
    
    return {
        'annotation_type': label_name,
        'kappa': kappa,
        'interpretation': interpretation,
        'agreement_percentage': agreement_pct,
        'agreement_count': agreement_count,
        'total_count': total_count,
        'confusion_matrix': cm.tolist(),
        'labels': unique_labels
    }

def analyze_disagreements(df1, df2, column1, column2):
    """
    Identify and analyze cases where annotators disagree.
    
    Args:
        df1: DataFrame for annotator 1
        df2: DataFrame for annotator 2
        column1: Column name in df1
        column2: Column name in df2
        
    Returns:
        DataFrame: Comments where annotators disagree
    """
    # Standardize labels for comparison
    labels1 = standardize_labels(df1[column1])
    labels2 = standardize_labels(df2[column2])
    
    # Find disagreements
    disagreements_mask = labels1 != labels2
    
    if disagreements_mask.any():
        disagreements = pd.DataFrame({
            'comment_id': df1.loc[disagreements_mask, 'comment_id'],
            'comment_text': df1.loc[disagreements_mask, 'comment_text'],
            'annotator1': labels1[disagreements_mask].values,
            'annotator2': labels2[disagreements_mask].values,
            'annotator1_confidence': df1.loc[disagreements_mask, 'manual confidence 0 to 1'].values,
            'annotator2_confidence': df2.loc[disagreements_mask, 'manual confidence 0 to 1'].values
        })
        return disagreements
    else:
        return pd.DataFrame()

def main():
    """
    Main function to calculate inter-annotator agreement.
    """
    # Set up paths
    base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
    annotations_path = base_path / "data" / "annotations" / "samples"
    results_path = base_path / "results" / "analysis" / "annotation_evaluation"
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print("Loading annotation files...")
    df1, df2 = load_annotations(annotations_path)
    
    # Ensure same number of annotations
    assert len(df1) == len(df2), f"Mismatch in annotation counts: {len(df1)} vs {len(df2)}"
    print(f"Loaded {len(df1)} annotations from each annotator")
    
    # Dictionary to store all results
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'total_annotations': len(df1),
        'metrics': {}
    }
    
    # Calculate Cohen's Kappa for each annotation type
    print("\nCalculating Cohen's Kappa scores...")
    print("-" * 60)
    
    # 1. Sentiment Agreement
    sentiment1 = standardize_labels(df1['manual sentiment'])
    sentiment2 = standardize_labels(df2['manual sentiment'])
    sentiment_results = calculate_kappa(sentiment1, sentiment2, 'Sentiment')
    results['metrics']['sentiment'] = sentiment_results
    
    print(f"Sentiment Agreement:")
    print(f"  Cohen's Kappa: {sentiment_results['kappa']:.3f} ({sentiment_results['interpretation']})")
    print(f"  Raw Agreement: {sentiment_results['agreement_percentage']:.1f}% ({sentiment_results['agreement_count']}/{sentiment_results['total_count']})")
    
    # 2. Learning Journey Agreement
    journey1 = standardize_labels(df1['manual learning journey'])
    journey2 = standardize_labels(df2['manual learning journey'])
    journey_results = calculate_kappa(journey1, journey2, 'Learning Journey')
    results['metrics']['learning_journey'] = journey_results
    
    print(f"\nLearning Journey Agreement:")
    print(f"  Cohen's Kappa: {journey_results['kappa']:.3f} ({journey_results['interpretation']})")
    print(f"  Raw Agreement: {journey_results['agreement_percentage']:.1f}% ({journey_results['agreement_count']}/{journey_results['total_count']})")
    
    # 3. Transition Agreement
    transition1 = standardize_labels(df1['manual has transition'])
    transition2 = standardize_labels(df2['manual has transition'])
    transition_results = calculate_kappa(transition1, transition2, 'Has Transition')
    results['metrics']['has_transition'] = transition_results
    
    print(f"\nHas Transition Agreement:")
    print(f"  Cohen's Kappa: {transition_results['kappa']:.3f} ({transition_results['interpretation']})")
    print(f"  Raw Agreement: {transition_results['agreement_percentage']:.1f}% ({transition_results['agreement_count']}/{transition_results['total_count']})")
    
    # Analyze disagreements
    print("\n" + "-" * 60)
    print("Analyzing disagreements...")
    
    # Sentiment disagreements
    sentiment_disagreements = analyze_disagreements(df1, df2, 'manual sentiment', 'manual sentiment')
    if not sentiment_disagreements.empty:
        print(f"\nSentiment disagreements: {len(sentiment_disagreements)} cases")
        sentiment_disagreements.to_csv(
            results_path / f"sentiment_disagreements_{results['timestamp']}.csv",
            index=False
        )
    
    # Learning journey disagreements
    journey_disagreements = analyze_disagreements(df1, df2, 'manual learning journey', 'manual learning journey')
    if not journey_disagreements.empty:
        print(f"Learning journey disagreements: {len(journey_disagreements)} cases")
        journey_disagreements.to_csv(
            results_path / f"journey_disagreements_{results['timestamp']}.csv",
            index=False
        )
    
    # Transition disagreements
    transition_disagreements = analyze_disagreements(df1, df2, 'manual has transition', 'manual has transition')
    if not transition_disagreements.empty:
        print(f"Transition disagreements: {len(transition_disagreements)} cases")
        transition_disagreements.to_csv(
            results_path / f"transition_disagreements_{results['timestamp']}.csv",
            index=False
        )
    
    # Save results to JSON
    results_file = results_path / f"inter_annotator_agreement_{results['timestamp']}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "-" * 60)
    print(f"Results saved to: {results_path}")
    print(f"  - Agreement metrics: {results_file.name}")
    if not sentiment_disagreements.empty:
        print(f"  - Sentiment disagreements: sentiment_disagreements_{results['timestamp']}.csv")
    if not journey_disagreements.empty:
        print(f"  - Journey disagreements: journey_disagreements_{results['timestamp']}.csv")
    if not transition_disagreements.empty:
        print(f"  - Transition disagreements: transition_disagreements_{results['timestamp']}.csv")
    
    # Generate summary report
    report_lines = [
        "INTER-ANNOTATOR AGREEMENT REPORT",
        "=" * 60,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total annotations: {results['total_annotations']}",
        "",
        "COHEN'S KAPPA SCORES:",
        "-" * 40,
        f"Sentiment:         {sentiment_results['kappa']:.3f} ({sentiment_results['interpretation']})",
        f"Learning Journey:  {journey_results['kappa']:.3f} ({journey_results['interpretation']})",
        f"Has Transition:    {transition_results['kappa']:.3f} ({transition_results['interpretation']})",
        "",
        "RAW AGREEMENT RATES:",
        "-" * 40,
        f"Sentiment:         {sentiment_results['agreement_percentage']:.1f}%",
        f"Learning Journey:  {journey_results['agreement_percentage']:.1f}%",
        f"Has Transition:    {transition_results['agreement_percentage']:.1f}%",
        "",
        "INTERPRETATION GUIDE (Landis & Koch, 1977):",
        "-" * 40,
        "< 0.00:  Poor (less than chance)",
        "0.00-0.20: Slight agreement",
        "0.21-0.40: Fair agreement",
        "0.41-0.60: Moderate agreement",
        "0.61-0.80: Substantial agreement",
        "0.81-1.00: Almost perfect agreement"
    ]
    
    report_file = results_path / f"agreement_report_{results['timestamp']}.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("\n" + '\n'.join(report_lines))

if __name__ == "__main__":
    main()