# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic research project (MSc Data Science thesis) analyzing mathematics education sentiment on YouTube using BERTopic and BERT models. The project employs sophisticated NLP techniques to understand user engagement with mathematical content through topic modeling and sentiment analysis.

## Core Commands

### Data Collection Pipeline

```bash
# Collect video metadata (respects API quotas)
python src/data_collection/daily_collector.py

# Collect comments (two versions available)
python src/data_collection/comment_collector.py
# Enhanced version:
python src/data_collection/comment_collector2.py
```

### Data Processing Pipeline

```bash
# Filter videos (expects ~96.7% retention rate)
python src/data_processing/video_filtering_pipeline.py

# Filter comments (expects ~44.2% retention rate)
python src/data_processing/comment_filtering_pipeline.py
```

### Analysis Commands

```bash
# Run BERTopic models (per-query approach)
python src/models/bertopic/per_query_hdbscan.py
python src/models/bertopic/variable_k_optimized.py

# Run sentiment analysis (various models with different performance)
python src/models/sentiment/variable_k_sentiment_analysis.py  # 37% accuracy (Twitter-trained, problematic aggregation)
python src/models/sentiment/variable_k_sentence_analysis_xlm_roberta.py  # 74.5% accuracy (YouTube-trained, same aggregation)
python src/analysis/hdbscan_sentiment_analysis.py
python src/models/sentiment/xlm_roberta_complete_dataset_analysis.py  # 74% accuracy (YouTube-trained, enhanced aggregation)

# Generate coherence analysis
python src/analysis/coherence/coherence_analysis.py

# Performance comparisons
python src/analysis/performance/bertopic_model_comparison.py

# Generate visualizations
python src/visualization/create_analysis_plots.py
python src/visualization/bertopic_comparison_charts.py

# Keyword Analysis (NEW - Added August 2025)
python src/analysis/topic_labeling/keyword_frequency_analyzer.py  # Keyword frequencies within topics
python src/analysis/topic_labeling/keyword_cooccurrence_analyzer.py  # Co-occurrence patterns ("girl math" analysis)
python src/analysis/simple_hierarchical_topics.py  # Query relationships per topic number
python src/analysis/keyword_hierarchy_tree.py  # Semantic keyword hierarchy tree

# Advanced Visualizations (NEW - Added August 2025)
python src/visualization/simple_keyword_network_viz.py  # Simplified keyword networks
python src/visualization/simple_variable_k_distance_map.py  # BERTopic-style intertopic distance maps

# Model Evaluation & Comparison (NEW - Added August 2025)
python src/analysis/model_evaluation/youtube_bert_agreement_analysis.py  # Independent YouTube-BERT evaluation (73% accuracy)
python src/analysis/model_evaluation/xlm_roberta_clean_agreement_analysis.py  # XLM-RoBERTa evaluation framework
```

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API access (REQUIRED)
cp config/config_template.py config/config.py
# Edit config/config.py with YouTube API key
```

## Architecture & Design Patterns

### Pipeline Architecture

The project follows a sequential data science pipeline:

1. **Data Collection** → YouTube API integration with quota management
2. **Data Processing** → Quality-focused filtering preserving educational content
3. **Topic Modeling** → Per-query BERTopic approach (83% noise reduction vs baseline)
4. **Sentiment Analysis** → BERT-based models for comment sentiment
5. **Validation** → Coherence analysis and statistical performance metrics

### Key Methodological Innovations

- **Per-Query BERTopic**: Processes each mathematical topic separately (8.72% noise vs 52.43% baseline)
- **Variable K-Means Clustering**: Dynamically optimizes K per query based on silhouette scores
- **Quality Filtering Pipeline**: Multi-stage validation ensuring educational content focus
- **Domain Training Supremacy Discovery** (NEW - August 16, 2025): Domain-specific training provides 36-37.5% improvement, 24x more impactful than model architecture choice
- **Educational Sentiment Challenge**: Universal model weakness on neutral educational content (20-23% misclassification rate across all models)
- **Sentiment Aggregation Discovery**: Simple majority voting outperforms complex weighting schemes (74% vs 37% accuracy)
- **Learning Journey Detection**: Identifies negative-to-positive progression patterns in educational comments (88% accuracy with XLM-RoBERTa Enhanced)
- **Keyword Hierarchy Analysis** (NEW - August 2025): Semantic reduction from 1,270 keywords to 8 parent categories
- **Co-occurrence Pattern Discovery** (NEW - August 2025): Cultural phenomena identification ("girl math" 3.2x more frequent than "boy math")
- **Simplified Network Analysis** (NEW - August 2025): Transparent frequency-based centrality instead of complex algorithms

### File Organization Patterns

- Model implementations use class-based structure with clear initialization and processing methods
- Analysis scripts follow functional approach with modular components
- All scripts include comprehensive docstrings explaining methodology
- Results are timestamped and saved to organized directory structure

### Data Flow

```
raw data → filtered data → embeddings → topic models → sentiment scores → visualizations
         ↓                ↓            ↓              ↓                 ↓
   data/raw/      data/processed/  models/      results/tables/  results/figures/
 (excluded)        (INCLUDED)                        ↓                 ↓
                                              keyword analysis → semantic hierarchies
                                                     ↓                 ↓
                                            co-occurrence maps → network visualizations
```

**Included Processed Datasets**:

- `data/processed/comments_complete_filtered_20250720_224959.csv` - Anonymized filtered comments
- `data/processed/videos_gradual_complete_filtered_20250720_223652.csv` - Filtered video metadata

## Critical Implementation Details

### BERTopic Configuration

- Embedding Model: `all-MiniLM-L6-v2` (SentenceTransformer)
- Clustering: Variable K-means with dynamic K optimization
- Min Topic Size: 10 documents
- N-gram Range: (1, 3) for topic representation

### Keyword Analysis Framework (NEW - August 2025)

**Hierarchy Structure**: 1,270 unique keywords → 8 semantic groups → 2 mega-categories
- **Mathematics Core** (323 occurrences): math → maths → mathematics → algebra → calculus → geometry
- **Appreciation** (336 occurrences): thank → thanks → great → amazing → love
- **Content** (160 occurrences): video → videos → channel → series
- **Question** (152 occurrences): question → answer → problem → solve
- **Education** (142 occurrences): exam → school → test → class
- **Learning** (144 occurrences): understand → teacher → learn → explanation
- **Help** (70 occurrences): helpful → help → helped
- **Difficulty** (46 occurrences): easy → hard → basic → difficult

**Co-occurrence Analysis**: Identifies compound phrases and cultural phenomena
- "girl math": 180 occurrences vs "boy math": 56 occurrences (3.2x ratio)
- Primary co-occurrences: "math" + "thank" (512), "video" + "thank" (547)

**Network Centrality**: "math" identified as structural center (0.9845 centrality score)

### Sentiment Analysis Models & Performance

**Key Finding**: Domain training >>> Aggregation method > Model architecture (August 2025 analysis)

#### Critical Discovery: Domain Training Impact (August 16, 2025 Analysis)
**Domain training provides 36-37.5% improvement** - strongest empirical finding
- Domain training: +36-37.5 percentage points improvement
- Aggregation method: +1 percentage point improvement  
- Model architecture: +1.5 percentage points improvement
- **Domain training is 24x more impactful** than model architecture choices

#### Four-Model Comprehensive Comparison (200 manually annotated samples):
**Using Identical Aggregation Method:**
- **Twitter-RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`): **37% accuracy** ❌
  - Twitter-trained, systematic bias toward negative sentiment
  - Over-predicts negative (57 vs 10 actual) - major educational content misclassification
- **YouTube-BERT** (`rahulk98/bert-finetuned-youtube_sentiment_analysis`): **73% accuracy** ✅
  - YouTube-trained, +36 percentage points over Twitter model
  - Better balanced predictions, closer to human distribution
- **XLM-RoBERTa Variable K** (`AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual`): **74.5% accuracy** ✅
  - YouTube-trained, +37.5 percentage points over Twitter model
  - Best overall performance with identical aggregation
- **XLM-RoBERTa Enhanced**: **74% accuracy** ✅
  - YouTube-trained with improved aggregation, minimal gain over Variable K

#### Learning Journey Detection Performance:
- **XLM-RoBERTa Enhanced**: 88% accuracy (best precision, least over-prediction)
- **YouTube-BERT**: 79% accuracy
- **Twitter-RoBERTa**: 56% accuracy (massive over-prediction: 77 vs 13 actual)

#### Critical Aggregation Differences:
- **Successful approach (Enhanced models, 74%)**: Simple majority voting, moderate confidence adjustments
- **Failed approach (Variable K Twitter-RoBERTa, 37%)**: Complex final-third weighting, rigid learning journey interpretation (auto-assigns 0.9 confidence positive)

#### Duplicate Handling Decision:
- All models preserve duplicate comments (e.g., "thanks a lot" × 529)
- Rationale: Each duplicate represents genuine user sentiment expression
- Natural frequency distribution provides accurate sentiment prevalence

### Performance Thresholds

- Video Duration: 60-3600 seconds
- Comment Length: 10-2000 characters
- Language Confidence: >0.6
- Non-Latin Script: <5%
- Statistical Improvement Threshold: 10% for K optimization

### API Quota Management

- MAX_VIDEOS_PER_QUERY: 1000
- MAX_COMMENTS_PER_VIDEO: 100
- QUERIES_PER_DAY: 100
- Implements exponential backoff for rate limiting

## Working with the Codebase

### Adding New Analysis

1. Create new script in appropriate `src/analysis/` subdirectory
2. Follow existing import patterns and class structure
3. Use timestamps for output files: `datetime.now().strftime("%Y%m%d_%H%M%S")`
4. Save results to `results/` with clear subdirectory structure

### Keyword Analysis Workflow (NEW - August 2025)

1. **Frequency Analysis**: Use `keyword_frequency_analyzer.py` for empirical keyword counts within topics
2. **Co-occurrence Detection**: Use `keyword_cooccurrence_analyzer.py` for phrase pattern discovery
3. **Network Visualization**: Use `simple_keyword_network_viz.py` for connection analysis (simplified approach preferred)
4. **Hierarchical Structure**: Use `keyword_hierarchy_tree.py` for semantic parent-child relationships
5. **Intertopic Mapping**: Use `simple_variable_k_distance_map.py` for BERTopic-style visualizations

**Key Principle**: Prefer simplified, frequency-based approaches over complex algorithms (PageRank, Louvain) for thesis transparency

### Modifying Topic Models

1. Check `src/models/bertopic/` for existing implementations
2. Maintain compatibility with downstream sentiment analysis
3. Preserve topic_info DataFrame structure for visualization scripts
4. Document any hyperparameter changes in docstrings

### Processing New Data

1. Ensure config/config.py has valid YouTube API key
2. Run collection scripts respecting quota limits
3. Verify filtering thresholds match research requirements
4. Check retention rates align with expected values

## Research Context

This is academic research code prioritizing:

- **Reproducibility**: Detailed methodology documentation
- **Statistical Rigor**: Validation through coherence analysis
- **Ethical Compliance**: GDPR-compliant data handling
- **Methodological Innovation**: Novel per-query approach

No traditional testing framework exists; validation occurs through:

- Manual annotation samples
- Statistical coherence metrics
- Cross-validation within models
- Performance comparison analysis

## Guidelines for AI Assistant

- You are supporting a graduate student prepare their thesis in fulfilment of a MSc Data Science
- Assume you are an erudite tutor in data science, machine learning and the social sciences.
- Do not be sycophantic; provide fair, honest feedback
- I am aiming for a grade of 'Distinction' or 80%+ on this ERP Project. Help me to meet this standard of quality
- Do not make any changes until you have 95% confidence that you know what to build. Ask follow-up questions until you have that confidence
- Keep scripts and code as simple and efficient as possible
- Do not use any emojis in scripts
- Do not be deceptive in your responses. If you do not find references in the files when prompted, do not modify any working scripts until you ask for clarification until evidence is found

## Thesis Evidence Cross-Reference

For detailed mapping between thesis claims and repository evidence, see `/Users/siradbihi/Desktop/MScDataScience/ERP_YouTube_Maths_Sentiments/THESIS_EVIDENCE_CROSSREF.md`. This document provides:
- Line-by-line evidence for all thesis claims
- Exact file paths and metrics for reproducibility
- Implementation details for all methodological innovations
- Statistical validation results with p-values
- Complete command sequence for pipeline reproduction

## Mandatory Evidence Requirements

- NEVER make claims without citing specific files and line numbers
- When confused about data sources, STOP and ask for clarification
- Before any analysis, state: "Based on [specific file], I can confirm X. I cannot confirm Y without additional evidence."
- If I cannot find evidence for a claim, I must say "I do not have evidence for this statement"
- NEVER fill gaps with speculation - explicitly state when inferring vs documenting facts

## Confidence Checks

- Before proceeding with any task, I must explicitly state my confidence level (e.g., "I am 95% confident" vs "I am speculating")
- If confidence < 95%, I must ask clarifying questions immediately
- I must distinguish between "what the data shows" vs "what I think it means"
- When uncertain, ask for user guidance rather than making assumptions

## Auto-correction Triggers

- If I use phrases like "this shows" or "the analysis reveals" - I must immediately cite the specific file and line numbers
- If I make comparative statements - I must verify I'm comparing the same metrics from the same source files
- If I present conclusions - I must separate documented facts from inferences with clear language
- If I contradict previous statements - I must acknowledge the contradiction and trace the source of confusion
- When accuracy figures are mentioned - I must identify the exact source files that generated those numbers

## Response Quality Standards

- State confidence level at the beginning of responses
- Cite file paths and line numbers for all factual claims
- Use precise language: "Based on file X:Y, I can confirm..." vs "This suggests..." 
- When speculating, explicitly label it as speculation
- Ask clarifying questions when evidence is missing rather than guessing
- Acknowledge mistakes immediately and trace their source

## Objective

-Consult these guidance documents to inform your support:

    - /Users/siradbihi/Desktop/MScDataScience/ERP Maths Sentiments/SAN-14151162-project-plan copy.pdf
    - /Users/siradbihi/Desktop/MScDataScience/Semester 2/DATA72002 Extended Research Project/DSM ERP Handbook 24-25 v2.1.pdf
    - /Users/siradbihi/Desktop/MScDataScience/Semester 2/DATA72002 Extended Research Project/DSM ERP intro slides 2025.pdf>
    - /Users/siradbihi/Desktop/MScDataScience/Semester 2/DATA72002 Extended Research Project/Reproducibility class v 2025(1).pdf>
    - /Users/siradbihi/Desktop/MScDataScience/Semester 2/DATA72002 Extended Research Project/SAN6 Perceptions of Mathematics in (Social) Media - Proposal.pdf>
