# Methodology

## Research Design

### Overview

This research employs a computational approach combining:

- topic modelling and sentiment classification
- Quality-focused data filtering for improved model performance
- Per-query optimization strategy for topic discovery

## Data Collection

### YouTube API v3

- **Search Queries**: 82 mathematics-related terms
- **Video Metadata**: Title, description, statistics
- **Comments**: Top-level and replies
- **Period**: [Collection dates]

### Sampling Strategy

- Maximum 1000 videos per query
- Maximum 100 comments per video
- English language content only

## Data Processing Pipeline

### Stage 1: Video Filtering

1. **Category Filtering**: Remove non-educational categories
2. **Keyword Validation**: Mathematical term presence
3. **Script Detection**: Latin script only (25+ scripts filtered)
4. **Quality Checks**: Duration, engagement metrics

### Stage 2: Comment Filtering

1. **Language Detection**: English with confidence > 0.6
2. **Length Validation**: 10-2000 characters
3. **Spam Removal**: Repetitive content detection
4. **Educational Relevance**: Mathematical context validation

## Analysis Methods

### Topic Modelling (BERTopic)

#### Innovation: Per-Query Approach

Instead of applying BERTopic to the entire dataset:

1. Create separate models for each search query
2. Optimize HDBSCAN parameters per query
3. Merge and analyze cross-query patterns

#### Configuration

- Embedding Model: all-MiniLM-L6-v2
- Clustering: HDBSCAN with adaptive parameters
- Dimensionality Reduction: UMAP
- Topic Representation: c-TF-IDF

### Sentiment Analysis

- Model: BERT-based classifier
- Classes: Positive, Neutral, Negative
- Validation: Manual annotation subset

### Coherence Evaluation

- Metrics: C_V, UMass
- Window Size: 10
- Reference Corpus: Processed comments

## Quality Assurance

### Filtering Effectiveness

- Video Retention: 96.7%
- Comment Retention: 44.2%
- Focus: Educational mathematics content

### Performance Metrics

- BERTopic Noise: 8.72% (per-query)
- Baseline Comparison: 52.43% (whole-dataset)
- Improvement: 83% noise reduction

### Validation Methods

- Silhouette scores for clustering
- Coherence metrics for topic modelling
- Inter-annotator agreement for annotations

## Ethical Considerations

### Privacy Protection

- User anonymization
- No personal data retention
- Aggregate reporting only

### Compliance

- YouTube Terms of Service
- GDPR requirements
- Institutional ethics approval

## Limitations

1. **Language Scope**: English content only
2. **Platform Specific**: YouTube only
3. **Time Period**: Snapshot, not longitudinal
4. **Selection Bias**: API search limitations

## Reproducibility

All code, configurations, and documentation provided for:

- Complete reproduction of results
- Adaptation to other domains
- Extension of methodology
