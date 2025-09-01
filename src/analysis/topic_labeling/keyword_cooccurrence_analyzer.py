#!/usr/bin/env python3
"""
Keyword Co-occurrence Analyzer for BERTopic Variable K Results
Identifies relationships between keywords that frequently appear together
MSc Data Science Thesis - Perceptions of Maths on YouTube
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from tqdm import tqdm

class KeywordCooccurrenceAnalyzer:
    def __init__(self):
        """Initialize keyword co-occurrence analyzer."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = Path("/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments")
        self.output_dir = self.base_path / "results" / "analysis" / "keyword_cooccurrence"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Keyword Co-occurrence Analyzer Initialized")
        print(f"Output directory: {self.output_dir}")
    
    def load_topic_documents(self):
        """Load the complete dataset with topic assignments."""
        comments_file = self.base_path / "results" / "models" / "bertopic_outputs" / "Optimised_Variable_K" / "merged_topic_info" / "optimised_variable_k_phase_4_20250722_224755" / "results" / "all_comments_with_topics.csv"
        
        if not comments_file.exists():
            comments_file = self.base_path / "results" / "models" / "bertopic_outputs" / "Optimised_Variable_K" / "all_comments_with_topics.csv"
        
        print(f"Loading comments with topic assignments...")
        df = pd.read_csv(comments_file)
        print(f"Loaded {len(df):,} comments")
        
        return df
    
    def extract_phrases_and_compounds(self, text, keywords):
        """Extract phrase patterns and compound terms from text."""
        text = text.lower()
        phrases_found = []
        
        # Define specific compound patterns to look for
        educational_compounds = [
            # Math-related compounds
            r'\bgirl\s+math\b', r'\bboy\s+math\b', r'\bmental\s+math\b',
            r'\bquick\s+math\b', r'\bbasic\s+math\b', r'\bhard\s+math\b',
            r'\beasy\s+math\b', r'\bfun\s+math\b', r'\badvanced\s+math\b',
            
            # Maths-related compounds  
            r'\bapplied\s+maths\b', r'\bpure\s+maths\b', r'\bgcse\s+maths\b',
            r'\blevel\s+maths\b', r'\bhigher\s+maths\b', r'\bcore\s+maths\b',
            
            # Teacher-related compounds
            r'\bmath\s+teacher\b', r'\bmaths\s+teacher\b', r'\bgreat\s+teacher\b',
            r'\bbest\s+teacher\b', r'\bgood\s+teacher\b',
            
            # Educational level compounds
            r'\bhigh\s+school\b', r'\bmiddle\s+school\b', r'\belementary\s+school\b',
            r'\bgrade\s+\d+\b', r'\byear\s+\d+\b',
            
            # Assessment compounds
            r'\bfinal\s+exam\b', r'\bmock\s+exam\b', r'\bpractice\s+test\b',
            r'\bpast\s+paper\b', r'\bexam\s+paper\b',
            
            # Subject-specific compounds
            r'\blinear\s+algebra\b', r'\bquadratic\s+equation\b', r'\bquadratic\s+formula\b',
            r'\bdata\s+science\b', r'\bcomputer\s+science\b',
            
            # Emotional/difficulty compounds
            r'\bmath\s+anxiety\b', r'\bmaths\s+anxiety\b', r'\bmath\s+phobia\b',
            r'\blearning\s+difficulty\b', r'\blearning\s+journey\b',
            
            # Gratitude compounds
            r'\bthank\s+you\b', r'\bthanks\s+so\b', r'\bso\s+helpful\b',
            r'\breally\s+helpful\b', r'\bvery\s+helpful\b'
        ]
        
        # Find compound patterns
        for pattern in educational_compounds:
            matches = re.findall(pattern, text)
            for match in matches:
                phrases_found.append(match)
        
        # Also look for any existing multi-word keywords from the topic
        for keyword in keywords:
            if ' ' in keyword and keyword in text:
                phrases_found.append(keyword)
        
        return phrases_found
    
    def calculate_cooccurrence_matrix(self, topic_documents, keywords_str):
        """Calculate co-occurrence patterns within topic documents."""
        if pd.isna(keywords_str):
            return {}, {}, {}
        
        # Parse keywords
        keywords = keywords_str.replace('[', '').replace(']', '').replace("'", '').split(',')
        keywords = [k.strip().lower() for k in keywords]
        
        # Initialize co-occurrence tracking
        pairwise_cooccurrence = defaultdict(int)
        phrase_patterns = defaultdict(int)
        keyword_context = defaultdict(list)
        
        # Process each document
        for doc in topic_documents:
            if pd.isna(doc):
                continue
            
            doc_lower = doc.lower()
            
            # Find which keywords appear in this document
            doc_keywords = []
            for keyword in keywords:
                if ' ' in keyword:
                    # Multi-word keyword - check exact match
                    if keyword in doc_lower:
                        doc_keywords.append(keyword)
                else:
                    # Single word - use word boundaries
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, doc_lower):
                        doc_keywords.append(keyword)
            
            # Calculate pairwise co-occurrence for this document
            if len(doc_keywords) >= 2:
                for i, kw1 in enumerate(doc_keywords):
                    for kw2 in doc_keywords[i+1:]:
                        pair = tuple(sorted([kw1, kw2]))
                        pairwise_cooccurrence[pair] += 1
            
            # Extract phrases and compounds
            phrases = self.extract_phrases_and_compounds(doc_lower, keywords)
            for phrase in phrases:
                phrase_patterns[phrase] += 1
            
            # Store context for each keyword (sample sentences)
            for keyword in doc_keywords[:3]:  # Limit to first 3 keywords per doc
                if len(keyword_context[keyword]) < 3:  # Limit context examples
                    sentences = re.split(r'[.!?]+', doc)
                    for sentence in sentences:
                        if keyword in sentence.lower() and len(sentence.strip()) > 10:
                            keyword_context[keyword].append(sentence.strip()[:100])
                            break
        
        return dict(pairwise_cooccurrence), dict(phrase_patterns), dict(keyword_context)
    
    def process_topics_cooccurrence(self):
        """Process all topics and calculate co-occurrence patterns."""
        # Load topic info
        topic_info_file = self.base_path / "results" / "models" / "bertopic_outputs" / "Optimised_Variable_K" / "merged_topic_info" / "merged_topic_info_sorted.csv"
        print(f"Loading topic info from: {topic_info_file}")
        topic_info_df = pd.read_csv(topic_info_file)  # Process all topics
        
        # Load comments with topics
        comments_df = self.load_topic_documents()
        
        print(f"\nProcessing {len(topic_info_df)} topics for co-occurrence analysis...")
        
        all_cooccurrences = defaultdict(int)
        all_phrases = defaultdict(int)
        topic_results = []
        
        for idx, row in tqdm(topic_info_df.iterrows(), total=len(topic_info_df), desc="Analyzing co-occurrences"):
            topic_num = row['Topic']
            query = row['query'] if 'query' in row else ''
            keywords = row['Representation'] if 'Representation' in row else ''
            doc_count = row['Count'] if 'Count' in row else 0
            
            # Get documents for this topic and query
            topic_docs = comments_df[
                (comments_df['topic'] == topic_num) & 
                (comments_df['search_query'] == query)
            ]['comment_text'].tolist() if 'search_query' in comments_df.columns else []
            
            if len(topic_docs) == 0:
                topic_docs = comments_df[comments_df['topic'] == topic_num]['comment_text'].tolist()
            
            # Calculate co-occurrence for this topic
            cooccur, phrases, context = self.calculate_cooccurrence_matrix(topic_docs, keywords)
            
            # Aggregate global patterns
            for pair, count in cooccur.items():
                all_cooccurrences[pair] += count
            
            for phrase, count in phrases.items():
                all_phrases[phrase] += count
            
            # Store topic-specific results
            top_pairs = sorted(cooccur.items(), key=lambda x: x[1], reverse=True)[:5]
            top_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)[:5]
            
            topic_results.append({
                'topic_number': topic_num,
                'query': query,
                'document_count': doc_count,
                'actual_doc_count': len(topic_docs),
                'top_cooccurrences': [f"{pair[0]} + {pair[1]} ({count})" for pair, count in top_pairs],
                'top_phrases': [f"{phrase} ({count})" for phrase, count in top_phrases],
                'cooccurrence_strength': sum(cooccur.values()),
                'phrase_diversity': len(phrases),
                'unique_pairs': len(cooccur)
            })
        
        return topic_results, dict(all_cooccurrences), dict(all_phrases)
    
    def create_cooccurrence_report(self, topic_results, global_cooccur, global_phrases):
        """Create comprehensive co-occurrence analysis report."""
        print("\nGenerating co-occurrence report...")
        
        # Sort global patterns by frequency
        top_cooccurrences = sorted(global_cooccur.items(), key=lambda x: x[1], reverse=True)[:30]
        top_phrases = sorted(global_phrases.items(), key=lambda x: x[1], reverse=True)[:30]
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("KEYWORD CO-OCCURRENCE ANALYSIS - BERTOPIC VARIABLE K")
        report_lines.append("=" * 80)
        report_lines.append(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Topics Analyzed: {len(topic_results)}")
        report_lines.append(f"Total Co-occurrence Patterns: {len(global_cooccur)}")
        report_lines.append(f"Total Phrase Patterns: {len(global_phrases)}")
        
        # Top keyword pairs
        report_lines.append("\n" + "=" * 50)
        report_lines.append("TOP 20 KEYWORD CO-OCCURRENCES (CORPUS-WIDE)")
        report_lines.append("=" * 50)
        for (kw1, kw2), count in top_cooccurrences[:20]:
            report_lines.append(f"{kw1} + {kw2}: {count} co-occurrences")
        
        # Top phrases and compounds
        report_lines.append("\n" + "=" * 50)
        report_lines.append("TOP 20 PHRASE PATTERNS")
        report_lines.append("=" * 50)
        for phrase, count in top_phrases[:20]:
            report_lines.append(f"'{phrase}': {count} occurrences")
        
        # Interesting semantic relationships
        report_lines.append("\n" + "=" * 50)
        report_lines.append("NOTABLE SEMANTIC RELATIONSHIPS")
        report_lines.append("=" * 50)
        
        # Group patterns by type
        math_compounds = [(p, c) for p, c in top_phrases if 'math' in p and ' ' in p]
        maths_compounds = [(p, c) for p, c in top_phrases if 'maths' in p and ' ' in p]
        teacher_compounds = [(p, c) for p, c in top_phrases if 'teacher' in p]
        educational_compounds = [(p, c) for p, c in top_phrases if any(word in p for word in ['school', 'grade', 'year', 'exam', 'test'])]
        
        if math_compounds:
            report_lines.append("\nMath-related Compound Terms:")
            for phrase, count in math_compounds[:8]:
                report_lines.append(f"  • '{phrase}': {count}")
        
        if maths_compounds:
            report_lines.append("\nMaths-related Compound Terms:")
            for phrase, count in maths_compounds[:8]:
                report_lines.append(f"  • '{phrase}': {count}")
        
        if teacher_compounds:
            report_lines.append("\nTeacher-related Compound Terms:")
            for phrase, count in teacher_compounds[:5]:
                report_lines.append(f"  • '{phrase}': {count}")
        
        if educational_compounds:
            report_lines.append("\nEducational Context Compounds:")
            for phrase, count in educational_compounds[:8]:
                report_lines.append(f"  • '{phrase}': {count}")
        
        # Topics with strongest co-occurrence patterns
        topic_results_sorted = sorted(topic_results, key=lambda x: x['cooccurrence_strength'], reverse=True)
        report_lines.append("\n" + "=" * 50)
        report_lines.append("TOPICS WITH STRONGEST KEYWORD RELATIONSHIPS")
        report_lines.append("=" * 50)
        
        for topic in topic_results_sorted[:10]:
            if topic['cooccurrence_strength'] > 0:
                report_lines.append(f"\n{topic['query']} Topic {topic['topic_number']} ({topic['document_count']} docs)")
                report_lines.append(f"  Co-occurrence strength: {topic['cooccurrence_strength']}")
                report_lines.append(f"  Unique keyword pairs: {topic['unique_pairs']}")
                if topic['top_cooccurrences']:
                    report_lines.append(f"  Top pairs: {' | '.join(topic['top_cooccurrences'][:3])}")
                if topic['top_phrases']:
                    report_lines.append(f"  Top phrases: {' | '.join(topic['top_phrases'][:3])}")
        
        # Save report
        report_file = self.output_dir / f"cooccurrence_analysis_report_{self.timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved co-occurrence report to: {report_file}")
        
        return report_file
    
    def create_cooccurrence_datasets(self, topic_results, global_cooccur, global_phrases):
        """Create datasets for visualization and further analysis."""
        print("\nCreating co-occurrence datasets...")
        
        # Global co-occurrence matrix
        cooccur_df = pd.DataFrame([
            {'keyword1': pair[0], 'keyword2': pair[1], 'cooccurrence_count': count}
            for pair, count in global_cooccur.items()
        ]).sort_values('cooccurrence_count', ascending=False)
        
        # Global phrase patterns
        phrases_df = pd.DataFrame([
            {'phrase': phrase, 'frequency': count}
            for phrase, count in global_phrases.items()
        ]).sort_values('frequency', ascending=False)
        
        # Topic-level co-occurrence summary
        topics_df = pd.DataFrame(topic_results)
        
        # Save datasets
        cooccur_file = self.output_dir / f"keyword_cooccurrences_{self.timestamp}.csv"
        phrases_file = self.output_dir / f"phrase_patterns_{self.timestamp}.csv"
        topics_file = self.output_dir / f"topic_cooccurrence_summary_{self.timestamp}.csv"
        
        cooccur_df.to_csv(cooccur_file, index=False)
        phrases_df.to_csv(phrases_file, index=False)
        topics_df.to_csv(topics_file, index=False)
        
        print(f"Saved co-occurrence datasets:")
        print(f"  • {cooccur_file}")
        print(f"  • {phrases_file}")
        print(f"  • {topics_file}")
        
        return cooccur_df, phrases_df, topics_df
    
    def run_analysis(self):
        """Execute complete co-occurrence analysis."""
        print("=" * 60)
        print("KEYWORD CO-OCCURRENCE ANALYSIS")
        print("=" * 60)
        
        # Process topics and calculate co-occurrences
        topic_results, global_cooccur, global_phrases = self.process_topics_cooccurrence()
        
        # Create analysis report
        report_file = self.create_cooccurrence_report(topic_results, global_cooccur, global_phrases)
        
        # Create datasets
        cooccur_df, phrases_df, topics_df = self.create_cooccurrence_datasets(
            topic_results, global_cooccur, global_phrases
        )
        
        print("\n" + "=" * 60)
        print("CO-OCCURRENCE ANALYSIS COMPLETE")
        print("=" * 60)
        
        print(f"\nKey Statistics:")
        print(f"  • Analyzed {len(topic_results)} topics")
        print(f"  • Found {len(global_cooccur)} unique keyword pairs")
        print(f"  • Identified {len(global_phrases)} phrase patterns")
        
        if global_cooccur:
            print(f"\nTop 5 Keyword Co-occurrences:")
            top_pairs = sorted(global_cooccur.items(), key=lambda x: x[1], reverse=True)[:5]
            for (kw1, kw2), count in top_pairs:
                print(f"  • {kw1} + {kw2}: {count} times")
        
        if global_phrases:
            print(f"\nTop 5 Phrase Patterns:")
            top_phrases = sorted(global_phrases.items(), key=lambda x: x[1], reverse=True)[:5]
            for phrase, count in top_phrases:
                print(f"  • '{phrase}': {count} times")
        
        print(f"\nOutput saved to: {self.output_dir}")
        
        return cooccur_df, phrases_df, topics_df

def main():
    """Main execution function."""
    analyzer = KeywordCooccurrenceAnalyzer()
    cooccur_df, phrases_df, topics_df = analyzer.run_analysis()

if __name__ == "__main__":
    main()