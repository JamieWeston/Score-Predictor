import os
from datetime import datetime

class Config:
    # API Configuration - your GitHub secrets will fill these automatically
    FOOTBALL_DATA_API_KEY = os.getenv('FOOTBALL_DATA_TOKEN')
    ODDS_API_KEY = os.getenv('ODDS_API_KEY')
    
    # Premier League ID for Football-Data.org API
    PREMIER_LEAGUE_ID = 2021
    
    # How many recent games to analyze
    RECENT_GAMES = 5
    HEAD_TO_HEAD_GAMES = 10
    
    # Confidence threshold for making betting tips
    CONFIDENCE_THRESHOLD = 60  # Only make tips when we're 60%+ confident
    
    # File paths
    PREDICTIONS_FILE = 'data/predictions.json'
    TIPS_FILE = 'data/tips.json'
    BACKUP_FIXTURES = 'data/fixtures.json'
