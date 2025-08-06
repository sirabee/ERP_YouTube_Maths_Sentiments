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
- **Variable K-Means Clustering**: Identified as top-performing clustering method through comprehensive comparison analysis
- **Quality-Focused Pipeline**: 96.7% video retention rate with 82 successful query models
- **Scale & Scope**: Analysis of 34,057 mathematics-focused comments from 3,897 educational videos across 82 mathematical topics
- **Rigorous Validation**: Coherence analysis, manual annotation, and statistical performance comparisons
- **Academic Excellence**: 5 comprehensive thesis support reports demonstrating research depth

## Repository Structure

```
src/
├── data_collection/     # YouTube API data collection scripts
├── data_processing/     # Video and comment filtering pipelines
├── models/
│   ├── bertopic/       # Per-query and Variable K clustering models
│   └── sentiment/      # BERT-based sentiment analysis models
├── analysis/
│   ├── coherence/      # Topic coherence validation
│   ├── performance/    # Model performance comparisons
│   └── topic_comparison/ # Cross-query topic analysis
├── visualization/      # Publication-ready figure generation
└── utils/             # Helper functions and utilities

data/
├── sample/            # Anonymized sample datasets
└── annotations/       # Manual annotation guidelines & samples

results/
├── figures/           # Performance dashboards and visualizations
├── tables/            # Performance metrics and statistical results
├── reports/           # Thesis support and methodology documentation
└── models/            # BERTopic outputs and comparisons

config/               # Configuration templates
docs/                 # Methodology and ethical documentation
scripts/              # Pipeline orchestration scripts
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

### Complete Pipeline

Run the full analysis pipeline:

```bash
# 1. Collect data
python src/data_collection/daily_collector.py
python src/data_collection/comment_collector.py

# 2. Process data
python src/data_processing/video_filtering_pipeline.py
python src/data_processing/comment_filtering_pipeline.py

# 3. Run analysis
python src/models/bertopic_model.py
python src/analysis/sentiment_analysis.py
```

### Individual Components

See `docs/REPRODUCIBILITY.md` for detailed step-by-step instructions.

## Data Privacy

In compliance with ethical research standards and GDPR:

- Raw YouTube data is **excluded** from this repository
- **Processed datasets with anonymized data are included** for transparency
- All user information has been anonymized (hashed IDs, removed channels)
- No API keys or credentials are included
- Included files:
  - `data/processed/comments_complete_filtered_20250720_224959.csv`
  - `data/processed/videos_gradual_complete_filtered_20250720_223652.csv`
- See `data/README.md` for detailed data information

## Results

### Performance Metrics

- **Video Filtering**: 3,897 videos retained (96.7% retention rate)
- **Comment Filtering**: 34,057 comments retained (44.2% retention rate)
- **Per-Query BERTopic**: 8.72% average noise across 82 query models
- **Baseline Comparison**: 52.43% noise (whole-dataset approach)
- **Performance Improvement**: 83% noise reduction using per-query methodology
- **Variable K-Means**: Identified as optimal clustering algorithm through comparative analysis

### Key Findings

- **Methodological Innovation**: Per-query topic modeling dramatically outperforms traditional approaches
- **Clustering Optimization**: Variable K-means clustering provides superior topic coherence
- **Quality-Focused Filtering**: Educational content validation significantly enhances model performance
- **Scale Validation**: Methodology successfully scales across 82 diverse mathematical topics
- **Coherence Validation**: Statistical analysis confirms superior topic quality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

_This repository is part of an MSc thesis submitted for the degree of Master of Science in Data Science._
