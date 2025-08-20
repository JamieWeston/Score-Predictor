#!/usr/bin/env python3
"""
Main script to generate football predictions
This runs automatically via GitHub Actions
"""

import sys
import os
from datetime import datetime

# Add the scripts directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_predictor import FootballPredictor

def main():
    """Main function to generate predictions"""
    print("=" * 60)
    print(f"🚀 FOOTBALL PREDICTOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = FootballPredictor()
        
        # Generate predictions
        predictions, tips = predictor.generate_all_predictions()
        
        if not predictions:
            print("❌ No predictions generated")
            sys.exit(1)
        
        # Save results
        success = predictor.save_predictions(predictions, tips)
        
        if success:
            print("=" * 60)
            print("✅ SUCCESS! Predictions updated successfully")
            print(f"📊 Generated {len(predictions)} match predictions")
            print(f"💡 Generated {len(tips)} betting tips")
            print("🌐 Your website will now show updated predictions")
            print("=" * 60)
        else:
            print("❌ Failed to save predictions")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# ====================================================================================================
# FILE 5: .github/workflows/predict.yml
# Create this file in the .github/workflows/ folder

name: Generate Football Predictions

on:
  schedule:
    # Run every day at 7 AM UTC (8 AM UK time)
    - cron: '0 7 * * *'
  workflow_dispatch: # Allows manual running

jobs:
  generate-predictions:
    runs-on: ubuntu-latest
    
    steps:
    - name: 📥 Checkout repository
      uses: actions/checkout@v4
    
    - name: 🐍 Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: 📦 Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: 🎯 Generate predictions
      env:
        FOOTBALL_DATA_TOKEN: ${{ secrets.FOOTBALL_DATA_TOKEN }}
        ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
      run: |
        cd scripts
        python generate_predictions.py
    
    - name: 📤 Commit and push changes
      run: |
        git config --local user.email "predictions@footballpredictor.com"
        git config --local user.name "Football Predictor Bot"
        git add data/predictions.json data/tips.json
        if git diff --staged --quiet; then
          echo "No changes to commit"
        else
          git commit -m "🤖 Auto-update predictions $(date +'%Y-%m-%d %H:%M:%S')"
          git push
        fi
