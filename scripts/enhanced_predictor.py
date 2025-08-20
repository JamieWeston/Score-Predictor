import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import requests
import json
import warnings
from config import Config

warnings.filterwarnings('ignore')

class FootballPredictor:
    def __init__(self):
        self.config = Config()
        self.models = {}
        
    def fetch_data_from_api(self):
        """Fetch latest data from Football API"""
        if not self.config.FOOTBALL_DATA_API_KEY:
            print("⚠️ No API key found, using backup data")
            return self.load_backup_data()
        
        headers = {'X-Auth-Token': self.config.FOOTBALL_DATA_API_KEY}
        url = f"https://api.football-data.org/v4/competitions/{self.config.PREMIER_LEAGUE_ID}/matches"
        
        try:
            print("📡 Fetching data from API...")
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                matches = self.parse_api_data(data['matches'])
                print(f"✅ Successfully fetched {len(matches)} matches")
                return matches
            elif response.status_code == 429:
                print("⏳ Rate limited, using backup data")
                return self.load_backup_data()
            else:
                print(f"❌ API Error {response.status_code}, using backup data")
                return self.load_backup_data()
                
        except Exception as e:
            print(f"❌ API request failed: {e}")
            return self.load_backup_data()
    
    def parse_api_data(self, matches):
        """Convert API data to our format"""
        parsed = []
        for match in matches:
            parsed_match = {
                'date': match['utcDate'][:10],
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'status': match['status']
            }
            
            # Add scores if finished
            if match['status'] == 'FINISHED':
                parsed_match['home_goals'] = match['score']['fullTime']['home']
                parsed_match['away_goals'] = match['score']['fullTime']['away']
            
            parsed.append(parsed_match)
        
        return pd.DataFrame(parsed)
    
    def load_backup_data(self):
        """Load backup data from local file"""
        try:
            with open(self.config.BACKUP_FIXTURES, 'r') as f:
                backup_data = json.load(f)
            
            # Convert to DataFrame if it's in JSON format
            if isinstance(backup_data, list):
                return pd.DataFrame(backup_data)
            else:
                # Handle different JSON structures
                matches = backup_data.get('matches', backup_data.get('data', []))
                return pd.DataFrame(matches)
                
        except FileNotFoundError:
            print("⚠️ No backup data found, creating minimal dataset")
            return self.create_minimal_dataset()
    
    def create_minimal_dataset(self):
        """Create minimal dataset for testing"""
        teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 
                'Tottenham', 'Newcastle', 'Brighton', 'West Ham', 'Aston Villa']
        
        # Create some fake historical data for the algorithm to work
        dates = pd.date_range(start='2024-01-01', end='2024-07-31', freq='3D')
        
        matches = []
        for i, date in enumerate(dates):
            if i < len(dates) - 10:  # Leave last 10 as upcoming
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
        
        # Add upcoming matches
        for i in range(10):
            date = datetime.now() + timedelta(days=i)
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            matches.append({
                'date': date.strftime('%Y-%m-%d'),
                'home_team': home_team,
                'away_team': away_team,
                'status': 'SCHEDULED'
            })
        
        return pd.DataFrame(matches)
    
    def calculate_team_form(self, team, all_matches, current_date, n_games=5):
        """Calculate recent form for a team"""
        team_matches = all_matches[
            ((all_matches['home_team'] == team) | (all_matches['away_team'] == team)) &
            (all_matches['status'] == 'FINISHED') &
            (pd.to_datetime(all_matches['date']) < pd.to_datetime(current_date))
        ].sort_values('date', ascending=False).head(n_games)
        
        if len(team_matches) == 0:
            return {'points': 0, 'goals_for': 1.0, 'goals_against': 1.0, 'games': 0}
        
        points = 0
        goals_for = 0
        goals_against = 0
        
        for _, match in team_matches.iterrows():
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
        return {
            'points': points,
            'goals_for': goals_for / games,
            'goals_against': goals_against / games,
            'games': games,
            'form_percentage': (points / (games * 3)) * 100 if games > 0 else 50
        }
    
    def calculate_head_to_head(self, home_team, away_team, all_matches, current_date):
        """Calculate head-to-head record"""
        h2h_matches = all_matches[
            (((all_matches['home_team'] == home_team) & (all_matches['away_team'] == away_team)) |
             ((all_matches['home_team'] == away_team) & (all_matches['away_team'] == home_team))) &
            (all_matches['status'] == 'FINISHED') &
            (pd.to_datetime(all_matches['date']) < pd.to_datetime(current_date))
        ].tail(self.config.HEAD_TO_HEAD_GAMES)
        
        if len(h2h_matches) == 0:
            return {'home_wins': 0, 'draws': 0, 'away_wins': 0, 'avg_goals': 2.5, 'btts_rate': 50}
        
        home_wins = draws = away_wins = total_goals = btts_count = 0
        
        for _, match in h2h_matches.iterrows():
            total_goals += match['home_goals'] + match['away_goals']
            
            if match['home_goals'] > 0 and match['away_goals'] > 0:
                btts_count += 1
            
            # From the perspective of our current home team
            if match['home_team'] == home_team:
                if match['home_goals'] > match['away_goals']:
                    home_wins += 1
                elif match['home_goals'] == match['away_goals']:
                    draws += 1
                else:
                    away_wins += 1
            else:  # Teams swapped in historical data
                if match['away_goals'] > match['home_goals']:
                    home_wins += 1
                elif match['away_goals'] == match['home_goals']:
                    draws += 1
                else:
                    away_wins += 1
        
        games = len(h2h_matches)
        return {
            'home_wins': home_wins,
            'draws': draws,
            'away_wins': away_wins,
            'avg_goals': total_goals / games,
            'btts_rate': (btts_count / games) * 100
        }
    
    def predict_match(self, home_team, away_team, match_date, all_matches):
        """Predict a single match outcome"""
        
        # Calculate team forms
        home_form = self.calculate_team_form(home_team, all_matches, match_date)
        away_form = self.calculate_team_form(away_team, all_matches, match_date)
        h2h = self.calculate_head_to_head(home_team, away_team, all_matches, match_date)
        
        # Simple prediction algorithm based on form and history
        home_strength = (home_form['form_percentage'] + 10) / 100  # Home advantage
        away_strength = away_form['form_percentage'] / 100
        
        # Expected goals based on recent form
        home_xg = max(0.5, home_form['goals_for'] * 1.1)  # Home advantage
        away_xg = max(0.5, away_form['goals_for'])
        
        # Adjust based on defensive records
        home_xg = home_xg / max(0.5, away_form['goals_against'])
        away_xg = away_xg / max(0.5, home_form['goals_against'])
        
        # Cap expected goals
        home_xg = min(home_xg, 4.0)
        away_xg = min(away_xg, 4.0)
        
        # Result probabilities
        total_strength = home_strength + away_strength
        if total_strength == 0:
            home_win_prob = draw_prob = away_win_prob = 33.33
        else:
            home_win_prob = (home_strength / total_strength) * 60 + 15  # 15-75% range
            away_win_prob = (away_strength / total_strength) * 60 + 15
            draw_prob = 100 - home_win_prob - away_win_prob
            
            # Ensure probabilities are reasonable
            home_win_prob = max(15, min(65, home_win_prob))
            away_win_prob = max(15, min(65, away_win_prob))
            draw_prob = max(15, min(40, draw_prob))
            
            # Normalize to 100%
            total_prob = home_win_prob + away_win_prob + draw_prob
            home_win_prob = (home_win_prob / total_prob) * 100
            away_win_prob = (away_win_prob / total_prob) * 100
            draw_prob = (draw_prob / total_prob) * 100
        
        # BTTS prediction
        btts_prob = min(85, max(15, (h2h['btts_rate'] + 
                                   ((home_form['goals_for'] + away_form['goals_for']) * 10)) / 2))
        
        # Over/Under 2.5
        expected_total_goals = home_xg + away_xg
        over_2_5_prob = min(85, max(15, (expected_total_goals / 3.5) * 100))
        
        # Top 3 scorelines
        scorelines = []
        for home_goals in range(6):
            for away_goals in range(6):
                prob = poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)
                scorelines.append({
                    'score': f"{home_goals}-{away_goals}",
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
                    'over': round(over_2_5_prob, 1),
                    'under': round(100 - over_2_5_prob, 1)
                },
                'top_scorelines': [
                    {'score': sl['score'], 'probability': round(sl['probability'], 1)}
                    for sl in top_scorelines
                ]
            },
            'team_info': {
                'home_form': f"{home_form['points']}/{home_form['games']*3} points",
                'away_form': f"{away_form['points']}/{away_form['games']*3} points",
                'h2h_games': h2h['home_wins'] + h2h['draws'] + h2h['away_wins']
            }
        }
    
    def generate_all_predictions(self):
        """Generate predictions for all upcoming matches"""
        print("🔄 Starting prediction generation...")
        
        # Get all match data
        all_matches = self.fetch_data_from_api()
        
        if len(all_matches) == 0:
            print("❌ No match data available")
            return [], []
        
        # Find upcoming matches
        upcoming_matches = all_matches[
            (all_matches['status'].isin(['SCHEDULED', 'TIMED'])) |
            (all_matches['status'].isna()) |
            (~all_matches.columns.isin(['home_goals', 'away_goals']).any())
        ].copy()
        
        # If no upcoming matches found, use last few rows as examples
        if len(upcoming_matches) == 0:
            print("⚠️ No upcoming matches found, using recent matches as examples")
            upcoming_matches = all_matches.tail(5).copy()
            upcoming_matches = upcoming_matches.drop(['home_goals', 'away_goals'], axis=1, errors='ignore')
        
        print(f"🎯 Generating predictions for {len(upcoming_matches)} matches...")
        
        predictions = []
        tips = []
        
        for _, match in upcoming_matches.iterrows():
            try:
                prediction = self.predict_match(
                    match['home_team'],
                    match['away_team'], 
                    match['date'],
                    all_matches
                )
                predictions.append(prediction)
                
                # Generate tip for this match
                tip = self.generate_tip_for_match(prediction)
                tips.append(tip)
                
            except Exception as e:
                print(f"❌ Failed to predict {match['home_team']} vs {match['away_team']}: {e}")
                continue
        
        print(f"✅ Generated {len(predictions)} predictions")
        return predictions, tips
    
    def generate_tip_for_match(self, prediction):
        """Generate betting tip based on prediction"""
        preds = prediction['predictions']
        best_tips = []
        
        # Check result prediction
        result_probs = preds['result']
        max_result_prob = max(result_probs.values())
        if max_result_prob >= self.config.CONFIDENCE_THRESHOLD:
            best_outcome = max(result_probs, key=result_probs.get)
            best_tips.append({
                'type': 'Match Result',
                'selection': best_outcome.replace('_', ' ').title(),
                'confidence': f"{max_result_prob}%",
                'reasoning': f"Strong form advantage"
            })
        
        # Check BTTS
        btts_yes = preds['btts']['yes']
        if btts_yes >= self.config.CONFIDENCE_THRESHOLD:
            best_tips.append({
                'type': 'Both Teams to Score',
                'selection': 'Yes',
                'confidence': f"{btts_yes}%",
                'reasoning': "Both teams scoring well recently"
            })
        elif (100 - btts_yes) >= self.config.CONFIDENCE_THRESHOLD:
            best_tips.append({
                'type': 'Both Teams to Score',
                'selection': 'No', 
                'confidence': f"{100 - btts_yes}%",
                'reasoning': "Defensive strength or poor attacking form"
            })
        
        # Check Over/Under 2.5
        over_prob = preds['over_under_2_5']['over']
        if over_prob >= self.config.CONFIDENCE_THRESHOLD:
            best_tips.append({
                'type': 'Total Goals',
                'selection': 'Over 2.5',
                'confidence': f"{over_prob}%",
                'reasoning': "High-scoring teams based on recent form"
            })
        elif (100 - over_prob) >= self.config.CONFIDENCE_THRESHOLD:
            best_tips.append({
                'type': 'Total Goals',
                'selection': 'Under 2.5',
                'confidence': f"{100 - over_prob}%",
                'reasoning': "Defensive teams or low-scoring trend"
            })
        
        return {
            'match': prediction['match'],
            'date': prediction['date'],
            'tips': best_tips if best_tips else [{
                'type': 'Recommendation',
                'selection': 'No Strong Tip',
                'confidence': '0%',
                'reasoning': 'No clear advantage detected'
            }]
        }
    
    def save_predictions(self, predictions, tips):
        """Save predictions and tips to JSON files"""
        try:
            # Save predictions
            with open(self.config.PREDICTIONS_FILE, 'w') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved predictions to {self.config.PREDICTIONS_FILE}")
            
            # Save tips  
            with open(self.config.TIPS_FILE, 'w') as f:
                json.dump(tips, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved tips to {self.config.TIPS_FILE}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to save files: {e}")
            return False
