# YouTube Mathematics Sentiment Analysis

## MSc Data Science Thesis Project

**Author**: 14151162  
**Institution**: The University of Manchester  
**Thesis Title**: Perceptions of Mathematics on YouTube: BERT-based Topic Modelling and Sentiment Analysis  
**Year**: 2025

## Overview

This repository contains the complete implementation for analyzing mathematics education content on YouTube using state-of-the-art NLP techniques. The project employs BERTopic for topic modeling and BERT-based models for sentiment analysis to understand how users engage with mathematical content on social media platforms.

### Key Achievements

- **Methodological Innovation**: Per-query BERTopic approach achieving 8.72% weighted average noise (83% improvement over baseline 52.43%)
- **Clustering Breakthrough**: Variable K-Means identified as optimal through comprehensive 81-algorithm comparison (0.425 silhouette score)
- **Domain Training Discovery**: YouTube-trained models outperform Twitter-trained by 36-37.5 percentage points (74% vs 37% accuracy)
- **Scale & Scope**: Analysis of 34,057 mathematics-focused comments from 3,897 educational videos across 82 mathematical topics
- **Learning Journey Detection**: 88% accuracy in identifying negative-to-positive sentiment progressions with XLM-RoBERTa
- **Keyword Hierarchy**: Semantic reduction from 1,270 keywords to 8 parent categories revealing discourse patterns

## Repository Structure

```
src/
├── data_collection/        # YouTube API data collection scripts
├── data_processing/        # Video and comment filtering pipelines
├── models/
│   ├── bertopic/          # Per-query and Variable K clustering models
│   └── sentiment/         # BERT-based sentiment analysis models
├── analysis/
│   ├── coherence/         # Topic coherence validation
│   ├── performance/       # Model performance comparisons
│   ├── model_evaluation/  # Sentiment model comparative analysis
│   ├── topic_labeling/    # Keyword analysis and hierarchies
│   └── annotation_evaluation/ # Inter-annotator agreement
├── visualization/         # Publication-ready figure generation
└── utils/                # Helper functions and utilities

data/
├── processed/            # Anonymized filtered datasets (INCLUDED)
├── annotations/          # Manual annotation samples
└── README.md            # Data documentation

results/
├── figures/             # Performance dashboards and visualizations
├── tables/              # Performance metrics and statistical results
├── models/              # BERTopic outputs and comparisons
└── visualizations/      # Interactive plots and network analyses

config/                  # Configuration templates
docs/                    # Methodology and ethical documentation
```

## Installation

### Prerequisites

- Python 3.8+
- 16GB RAM minimum
- GPU recommended for BERTopic
- YouTube Data API v3 key

### Setup

1. Clone the repository:

```bash
git clone [repository-url]
cd ERP_YouTube_Maths_Sentiments
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure API access:

```bash
cp config/config_template.py config/config.py
# Edit config.py with your YouTube API key
```

## Usage

### Quick Start - Analysis Only (Using Included Data)

```bash
# Run BERTopic Variable K modeling
python src/models/bertopic/variable_k_optimized.py

# Run sentiment analysis (XLM-RoBERTa)
python src/models/sentiment/xlm_roberta_clean_enhanced.py

# Generate visualizations
python src/visualization/create_analysis_plots.py
```

### Complete Pipeline (From Data Collection)

```bash
# 1. Collect data
python src/data_collection/daily_collector.py
python src/data_collection/comment_collector2.py

# 2. Process data
python src/data_processing/video_filtering_pipeline.py
python src/data_processing/comment_filtering_pipeline.py

# 3. Run topic modeling
python src/models/bertopic/variable_k_optimized.py

# 4. Run sentiment analysis
python src/models/sentiment/xlm_roberta_clean_enhanced.py

# 5. Generate analysis
python src/visualization/create_analysis_plots.py
python src/visualization/model_comparison_visualizations.py
```

### Advanced Analysis

```bash
# Keyword hierarchy analysis
python src/analysis/topic_labeling/keyword_frequency_analyzer.py
python src/analysis/topic_labeling/keyword_cooccurrence_analyzer.py
python src/analysis/keyword_hierarchy_tree.py

# Learning journey detection
python src/analysis/learning_journey_video_analysis.py

# Model performance comparison
python src/analysis/performance/bertopic_model_comparison.py
```

See `REPRODUCIBILITY.md` for detailed step-by-step instructions.

## Data Privacy

In compliance with ethical research standards and GDPR:

- Raw YouTube data is **excluded** from this repository
- **Processed datasets with anonymized data are included** for transparency
- All user information has been anonymized (hashed IDs, removed channels)
- No API keys or credentials are included
- Included files:
  - `data/processed/comments_complete_filtered_20250720_224959.csv` (34,057 comments)
  - `data/processed/videos_gradual_complete_filtered_20250720_223652.csv` (3,897 videos)
- See `data/README.md` for detailed data information

## Results

### Performance Metrics

#### Topic Modeling
- **Per-Query BERTopic**: 8.72% weighted average noise (83% improvement over baseline)
- **Variable K-Means**: 0.425 mean silhouette score (best of 81 algorithms tested)
- **Topics Generated**: 12 distinct topics across 378 query-topic combinations
- **Query Coverage**: 82/82 queries successfully modeled

#### Sentiment Analysis
- **XLM-RoBERTa Enhanced**: 74% accuracy (YouTube-trained)
- **YouTube-BERT**: 73% accuracy
- **Twitter-RoBERTa**: 37% accuracy (demonstrates domain training importance)
- **Learning Journey Detection**: 88% accuracy with XLM-RoBERTa

#### Data Retention
- **Video Filtering**: 3,897 videos retained (96.7% retention rate)
- **Comment Filtering**: 34,057 comments retained (44.2% retention rate)

### Key Findings

1. **Domain Training Supremacy**: YouTube-trained models provide 36-37.5 percentage point improvement over Twitter-trained models
2. **Aggregation Method Impact**: Simple majority voting outperforms complex weighting (74% vs 37% accuracy)
3. **Educational Content Challenge**: All models show 20-23% misclassification on neutral educational content
4. **Cultural Phenomena Detection**: "Girl math" appears 3.2x more frequently than "boy math" in discourse
5. **Keyword Semantic Structure**: 1,270 unique keywords reduce to 8 core categories (math, appreciation, content, question, education, learning, help, difficulty)

## Key Scripts Reference

### Core Commands (from CLAUDE.md)

```bash
# BERTopic models (per-query approach)
python src/models/bertopic/per_query_hdbscan.py
python src/models/bertopic/variable_k_optimized.py

# Sentiment analysis (various models)
python src/models/sentiment/xlm_roberta_clean_enhanced.py  # 74% accuracy (recommended)
python src/models/sentiment/variable_k_sentence_analysis_xlm_roberta.py  # 74.5% accuracy
python src/models/sentiment/per-sentence_analysis-variable-k-youtube.py  # 73% accuracy

# Generate coherence analysis
python src/analysis/coherence/coherence_analysis.py

# Performance comparisons
python src/analysis/performance/bertopic_model_comparison.py

# Keyword Analysis
python src/analysis/topic_labeling/keyword_frequency_analyzer.py
python src/analysis/topic_labeling/keyword_cooccurrence_analyzer.py
python src/analysis/simple_hierarchical_topics.py
python src/analysis/keyword_hierarchy_tree.py

# Advanced Visualizations
python src/visualization/simple_keyword_network_viz.py
python src/visualization/simple_variable_k_distance_map.py

# Model Evaluation & Comparison
python src/analysis/model_evaluation/youtube_bert_agreement_analysis.py
python src/analysis/model_evaluation/xlm_roberta_clean_agreement_analysis.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If using this code or methodology, please cite:

```
Author (2025). Perceptions of Mathematics on YouTube: 
BERT-based Topic Modelling and Sentiment Analysis. 
MSc Thesis, University of Manchester.
```

## Acknowledgments

This research was conducted as part of the MSc Data Science programme at The University of Manchester. Special thanks to the academic supervisors and the YouTube mathematics education community whose content made this analysis possible.

---

_This repository is part of an MSc thesis submitted for the degree of Master of Science in Data Science._