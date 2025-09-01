"""
Configuration Template for YouTube Mathematics Sentiment Analysis
Copy this file to config.py and add your actual values
"""

# YouTube API Configuration
YOUTUBE_API_KEY = "your_youtube_api_key_here"

# Data Collection Settings
MAX_VIDEOS_PER_QUERY = 1000
MAX_COMMENTS_PER_VIDEO = 100
QUERIES_PER_DAY = 100

# Search Queries (mathematical terms)
SEARCH_QUERIES = [
    "algebra tutorial",
    "calculus explained",
    "geometry proof",
    "statistics lesson",
    "mathematics education",
    # Add more queries as needed
]

# Filtering Thresholds
MIN_COMMENT_LENGTH = 10
MAX_COMMENT_LENGTH = 2000
MIN_VIDEO_DURATION = 60  # seconds
MAX_VIDEO_DURATION = 3600  # seconds
LANGUAGE_CONFIDENCE_THRESHOLD = 0.6
NON_LATIN_SCRIPT_THRESHOLD = 0.05

# Model Parameters
RANDOM_SEED = 42
BATCH_SIZE = 32
N_JOBS = -1  # Use all CPU cores
DEVICE = "cuda"  # or "cpu"

# BERTopic Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE = 10
N_GRAM_RANGE = (1, 3)

# Paths
BASE_DIR = "."
DATA_DIR = f"{BASE_DIR}/data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
RESULTS_DIR = f"{BASE_DIR}/results"
MODELS_DIR = f"{BASE_DIR}/models"
FIGURES_DIR = f"{RESULTS_DIR}/figures"
TABLES_DIR = f"{RESULTS_DIR}/tables"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "pipeline.log"
