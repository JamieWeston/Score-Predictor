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
