import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from scipy.stats import poisson

# Ensure data directory exists
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'scripts', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")

class SimpleFootballPredictor:
    def __init__(self):
        self.api_key = os.getenv('FOOTBALL_DATA_TOKEN')
        self.premier_league_id = 2021
        
    def fetch_data(self):
        """Fetch data from API or create sample data"""
        print("📡 Fetching match data...")
        
        if not self.api_key:
            print("⚠️ No API key found, creating sample data")
            return self.create_sample_data()
        
        try:
            headers = {'X-Auth-Token': self.api_key}
            url = f"https://api.football-data.org/v4/competitions/{self.premier_league_id}/matches"
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                matches = self.parse_api_data(data['matches'])
                print(f"✅ Fetched {len(matches)} matches from API")
                return matches
            else:
                print(f"⚠️ API returned {response.status_code}, using sample data")
                return self.create_sample_data()
                
        except Exception as e:
            print(f"⚠️ API error: {e}, using sample data")
            return self.create_sample_data()
    
    def parse_api_data(self, matches):
        """Parse API response"""
        parsed = []
        for match in matches:
            parsed_match = {
                'date': match['utcDate'][:10],
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'status': match['status']
            }
            
            if match['status'] == 'FINISHED' and match['score']['fullTime']:
                parsed_match['home_goals'] = match['score']['fullTime']['home']
                parsed_match['away_goals'] = match['score']['fullTime']['away']
            
            parsed.append(parsed_match)
        
        return pd.DataFrame(parsed)
    
    def create_sample_data(self):
        """Create sample data for testing"""
        print("🔧 Creating sample data...")
        
        teams = [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton & Hove Albion',
            'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich Town',
            'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
            'Newcastle United', 'Nottingham Forest', 'Southampton', 'Tottenham Hotspur',
            'West Ham United', 'Wolverhampton Wanderers'
        ]
        
        matches = []
        
        # Create historical matches (last 3 months)
        start_date = datetime.now() - timedelta(days=90)
        for i in range(200):  # 200 historical matches
            date = start_date + timedelta(days=i * 0.4)  # Spread over 80 days
            
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            matches.append({
                'date': date.strftime('%Y-%m-%d'),
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': np.random.poisson(1.3),
                'away_goals': np.random.poisson(1.1),
                'status': 'FINISHED'
            })
        
        # Create upcoming matches (next 2 weeks)
        for i in range(20):  # 20 upcoming matches
            date = datetime.now() + timedelta(days=i * 0.7)
            
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            matches.append({
                'date': date.strftime('%Y-%m-%d'),
                'home_team': home_team,
                'away_team': away_team,
                'status': 'SCHEDULED'
            })
        
        return pd.DataFrame(matches)
    
    def calculate_team_form(self, team, all_matches, current_date):
        """Calculate team form"""
        team_matches = all_matches[
            ((all_matches['home_team'] == team) | (all_matches['away_team'] == team)) &
            (all_matches['status'] == 'FINISHED') &
            (pd.to_datetime(all_matches['date']) < pd.to_datetime(current_date))
        ].sort_values('date', ascending=False).head(5)
        
        if len(team_matches) == 0:
            return {'points': 6, 'goals_for': 1.5, 'goals_against': 1.5}
        
        points = 0
        goals_for = 0
        goals_against = 0
        
        for _, match in team_matches.iterrows():
            if pd.isna(match.get('home_goals')) or pd.isna(match.get('away_goals')):
                continue
                
            if match['home_team'] == team:
                gf, ga = match['home_goals'], match['away_goals']
            else:
                gf, ga = match['away_goals'], match['home_goals']
            
            goals_for += gf
            goals_against += ga
            
            if gf > ga:
                points += 3
            elif gf == ga:
                points += 1
        
        games = len(team_matches)
        if games == 0:
            return {'points': 6, 'goals_for': 1.5, 'goals_against': 1.5}
            
        return {
            'points': points,
            'goals_for': goals_for / games,
            'goals_against': goals_against / games
        }
    
    def predict_match(self, home_team, away_team, match_date, all_matches):
        """Predict a single match"""
        
        home_form = self.calculate_team_form(home_team, all_matches, match_date)
        away_form = self.calculate_team_form(away_team, all_matches, match_date)
        
        # Calculate expected goals
        home_xg = max(0.5, home_form['goals_for'] * 1.15)  # Home advantage
        away_xg = max(0.5, away_form['goals_for'])
        
        # Adjust for opponent defense
        home_xg = home_xg * (2.0 / max(1.0, away_form['goals_against']))
        away_xg = away_xg * (2.0 / max(1.0, home_form['goals_against']))
        
        # Cap expected goals
        home_xg = min(home_xg, 3.5)
        away_xg = min(away_xg, 3.5)
        
        # Calculate result probabilities
        home_strength = home_form['points'] + 2  # Home advantage
        away_strength = away_form['points']
        total_strength = home_strength + away_strength
        
        if total_strength == 0:
            home_win_prob = draw_prob = away_win_prob = 33.33
        else:
            home_win_prob = max(20, min(70, (home_strength / total_strength) * 85))
            away_win_prob = max(15, min(65, (away_strength / total_strength) * 85))
            draw_prob = 100 - home_win_prob - away_win_prob
            
            # Ensure draw probability is reasonable
            if draw_prob < 15:
                draw_prob = 15
                remaining = 85
                home_win_prob = (home_win_prob / (home_win_prob + away_win_prob)) * remaining
                away_win_prob = remaining - home_win_prob
        
        # BTTS probability
        avg_goals_per_team = (home_xg + away_xg) / 2
        btts_prob = min(85, max(25, avg_goals_per_team * 25))
        
        # Over/Under 2.5
        total_expected = home_xg + away_xg
        over_prob = min(85, max(25, (total_expected / 3.0) * 100))
        
        # Top scorelines using Poisson
        scorelines = []
        for h in range(6):
            for a in range(6):
                prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                scorelines.append({
                    'score': f"{h}-{a}",
                    'probability': prob * 100
                })
        
        top_scorelines = sorted(scorelines, key=lambda x: x['probability'], reverse=True)[:3]
        
        return {
            'match': f"{home_team} vs {away_team}",
            'date': match_date,
            'predictions': {
                'result': {
                    'home_win': round(home_win_prob, 1),
                    'draw': round(draw_prob, 1),
                    'away_win': round(away_win_prob, 1)
                },
                'btts': {
                    'yes': round(btts_prob, 1),
                    'no': round(100 - btts_prob, 1)
                },
                'over_under_2_5': {
                    'over': round(over_prob, 1),
                    'under': round(100 - over_prob, 1)
                },
                'top_scorelines': [
                    {'score': sl['score'], 'probability': round(sl['probability'], 1)}
                    for sl in top_scorelines
                ]
            }
        }
    
    def generate_all_predictions(self):
        """Generate all predictions"""
        print("🎯 Generating predictions...")
        
        # Get match data
        all_matches = self.fetch_data()
        
        # Find upcoming matches
        upcoming = all_matches[
            ~all_matches.get('home_goals', pd.Series()).notna() |
            (all_matches['status'] == 'SCHEDULED')
        ].copy()
        
        if len(upcoming) == 0:
            # Use some recent matches as examples
            upcoming = all_matches.tail(5).copy()
        
        predictions = []
        tips = []
        
        for _, match in upcoming.head(10).iterrows():  # Limit to 10 matches
            try:
                prediction = self.predict_match(
                    match['home_team'],
                    match['away_team'],
                    match['date'],
                    all_matches
                )
                predictions.append(prediction)
                
                # Generate simple tip
                result_probs = prediction['predictions']['result']
                max_prob = max(result_probs.values())
                
                if max_prob > 55:
                    best_outcome = max(result_probs, key=result_probs.get)
                    tip_text = best_outcome.replace('_', ' ').title()
                else:
                    tip_text = "No Clear Favorite"
                
                tips.append({
                    'match': prediction['match'],
                    'date': prediction['date'],
                    'tip': tip_text,
                    'confidence': f"{max_prob:.1f}%"
                })
                
            except Exception as e:
                print(f"❌ Error predicting {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')}: {e}")
                continue
        
        return predictions, tips
    
    def save_files(self, predictions, tips):
        """Save prediction files"""
        try:
            # Ensure data directory exists
            os.makedirs('data', exist_ok=True)
            
            # Save predictions
            predictions_file = 'data/predictions.json'
            with open(predictions_file, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(predictions)} predictions to {predictions_file}")
            
            # Save tips
            tips_file = 'data/tips.json'
            with open(tips_file, 'w', encoding='utf-8') as f:
                json.dump(tips, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(tips)} tips to {tips_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save files: {e}")
            return False

def main():
    """Main function"""
    print("=" * 60)
    print(f"🚀 FOOTBALL PREDICTOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Create predictor
        predictor = SimpleFootballPredictor()
        
        # Generate predictions
        predictions, tips = predictor.generate_all_predictions()
        
        if not predictions:
            print("❌ No predictions generated")
            sys.exit(1)
        
        # Save files
        success = predictor.save_files(predictions, tips)
        
        if success:
            print("=" * 60)
            print("✅ SUCCESS! Files updated successfully")
            print(f"📊 Generated {len(predictions)} predictions")
            print(f"💡 Generated {len(tips)} tips")
            print("=" * 60)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
