
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

RAW_DATA_DIR = './raw_data'
DATA_DIR = './data'
FINAL_DATA_DIR = './final_data'
RESULTS_DIR = './results'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FINAL_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"TensorFlow version: {tf.__version__}")
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ============================================================
# PART 1: INITIAL DATA PREPROCESSING (Aadib's prep)
# ============================================================
def preprocess_season(raw_filename, season_tag):
    """Process a single season's raw CSV into position-specific clean files."""
    file_path = os.path.join(RAW_DATA_DIR, raw_filename)
    df = pd.read_csv(file_path)
    print(f"\n===== Processing season {season_tag} from {raw_filename} =====")
    print(f"Total raw rows: {len(df)}")

    features_to_keep = [
        'name', 'position', 'team', 'GW',
        'total_points', 'goals_scored', 'assists', 'minutes',
        'expected_goals', 'expected_assists',
        'fixture', 'was_home', 'clean_sheets'
    ]
    df_clean = df[features_to_keep].copy()
    df_clean['season'] = season_tag

    df_clean = df_clean.sort_values(by=['name', 'GW']).reset_index(drop=True)

    # Engineered form (4-game rolling average)
    df_clean['engineered_form'] = df_clean.groupby('name')['total_points'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )

    # Drop bench players
    rows_before = len(df_clean)
    df_clean = df_clean[df_clean['minutes'] > 0].copy()
    rows_after = len(df_clean)
    print(f"Removed {rows_before - rows_after} bench rows.")

    # Target: next gameweek points
    df_clean['Next_GW_Points'] = df_clean.groupby('name')['total_points'].shift(-1)
    df_clean = df_clean.dropna(subset=['Next_GW_Points']).copy()

    # Split by position
    positions = ['GK', 'DEF', 'MID', 'FWD']
    for pos in positions:
        df_pos = df_clean[df_clean['position'] == pos]
        out_path = os.path.join(DATA_DIR, f'{pos}_clean_{season_tag}.csv')
        df_pos.to_csv(out_path, index=False)

    print(f"Exported GK:{len(df_clean[df_clean['position']=='GK'])}, "
          f"DEF:{len(df_clean[df_clean['position']=='DEF'])}, "
          f"MID:{len(df_clean[df_clean['position']=='MID'])}, "
          f"FWD:{len(df_clean[df_clean['position']=='FWD'])} rows.")

def combine_seasons():
    """Merge season-specific clean files into all_seasons CSVs."""
    seasons = ['23_24', '24_25', '25_26']
    positions = ['GK', 'DEF', 'MID', 'FWD']
    for pos in positions:
        dfs = []
        for season in seasons:
            path = os.path.join(DATA_DIR, f'{pos}_clean_{season}.csv')
            df = pd.read_csv(path)
            if 'season' not in df.columns:
                df['season'] = season
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(['name', 'season', 'GW']).reset_index(drop=True)
        out_path = os.path.join(DATA_DIR, f'{pos}_all_seasons.csv')
        combined.to_csv(out_path, index=False)
    print("\n✅ Combined all seasons into *_all_seasons.csv")

print("\n" + "="*70)
print("STEP 1: INITIAL PREPROCESSING")
print("="*70)
preprocess_season('raw_23_24.csv', '23_24')
preprocess_season('raw_24_25.csv', '24_25')
preprocess_season('raw_25_26.csv', '25_26')
combine_seasons()

# ============================================================
# PART 2: FEATURE ENGINEERING (OPPONENT & LEAGUE CONTEXT)
# ============================================================
print("\n" + "="*70)
print("STEP 2: FEATURE ENGINEERING")
print("="*70)

# Team name mapping
team_name_map = {
    "West Ham": "West Ham United",
    "Nott'm Forest": "Nottingham Forest",
    "Newcastle": "Newcastle United",
    "Wolves": "Wolverhampton Wanderers",
    "Leeds": "Leeds United",
    "Brighton": "Brighton and Hove Albion",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Spurs": "Tottenham Hotspur",
    "Leicester": "Leicester City",
}
full_names = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Chelsea",
    "Crystal Palace", "Everton", "Fulham", "Ipswich Town", "Liverpool",
    "Manchester City", "Manchester United", "Newcastle United", "Nottingham Forest",
    "Southampton", "Tottenham Hotspur", "West Ham United", "Wolverhampton Wanderers",
    "Leicester City", "Brighton and Hove Albion"
]
for name in full_names:
    team_name_map[name] = name

team_id_to_name = {
    1: "Arsenal", 2: "Aston Villa", 3: "Bournemouth", 4: "Brentford",
    5: "Brighton and Hove Albion", 6: "Chelsea", 7: "Crystal Palace", 8: "Everton",
    9: "Fulham", 10: "Ipswich Town", 11: "Leicester City", 12: "Liverpool",
    13: "Manchester City", 14: "Manchester United", 15: "Newcastle United",
    16: "Nottingham Forest", 17: "Southampton", 18: "Tottenham Hotspur",
    19: "West Ham United", 20: "Wolverhampton Wanderers"
}

# Load baseline CSVs
positions = ['GK', 'DEF', 'MID', 'FWD']
raw_dfs = {}
for pos in positions:
    file_path = os.path.join(DATA_DIR, f'{pos}_all_seasons.csv')
    df = pd.read_csv(file_path)
    df['season'] = df['season'].str.replace('_', '-').apply(lambda x: f"20{x}" if len(x)==5 else x)
    df['team'] = df['team'].map(team_name_map).fillna(df['team'])
    raw_dfs[pos] = df
    print(f"  ✅ Loaded {pos}: {len(df):,} rows")

# Fetch fixtures from Vaastav
fixture_data = []
seasons_fixture = ['2023-24', '2024-25', '2025-26']
for season in seasons_fixture:
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/fixtures.csv"
    try:
        fixtures = pd.read_csv(url)
        fixtures['season'] = season
        fixtures.rename(columns={'event': 'GW'}, inplace=True)
        for _, row in fixtures.iterrows():
            if pd.notna(row['GW']):
                gw = int(row['GW'])
                fixture_data.append({
                    'season': season, 'GW': gw,
                    'team': row['team_h'], 'opponent': row['team_a'],
                    'fixture': row['id']
                })
                fixture_data.append({
                    'season': season, 'GW': gw,
                    'team': row['team_a'], 'opponent': row['team_h'],
                    'fixture': row['id']
                })
        print(f"  ✅ Loaded fixtures for {season}")
    except Exception as e:
        print(f"  ⚠️ Could not load {season}: {e}")

fixture_df = pd.DataFrame(fixture_data)
fixture_df['team'] = fixture_df['team'].map(team_id_to_name)
fixture_df['opponent'] = fixture_df['opponent'].map(team_id_to_name)
fixture_df = fixture_df.dropna()
fixture_df = fixture_df.drop_duplicates(subset=['season', 'fixture', 'team'])
print(f"  ✅ Created opponent lookup: {len(fixture_df):,} rows")

# Add opponent column and clean
for pos in positions:
    df = raw_dfs[pos]
    df = df.merge(fixture_df[['season', 'fixture', 'opponent']], on=['season', 'fixture'], how='inner')
    df = df[df['team'] != df['opponent']].copy()
    df = df.drop_duplicates(subset=['name', 'season', 'GW'], keep='first')
    raw_dfs[pos] = df
    print(f"  ✅ {pos}: {len(df):,} rows after cleaning")

all_players = pd.concat(raw_dfs.values(), ignore_index=True)

# Calculate real league positions (from goals)
team_goals = all_players.groupby(['season', 'GW', 'team'])['goals_scored'].sum().reset_index()
matches = fixture_df.merge(team_goals, on=['season', 'GW', 'team'], how='left')
matches.rename(columns={'goals_scored': 'team_goals'}, inplace=True)
matches = matches.merge(team_goals, left_on=['season', 'GW', 'opponent'], right_on=['season', 'GW', 'team'], how='left', suffixes=('', '_opp'))
matches.rename(columns={'goals_scored': 'opp_goals'}, inplace=True)
matches['team_goals'] = matches['team_goals'].fillna(0)
matches['opp_goals'] = matches['opp_goals'].fillna(0)
matches['match_points'] = matches.apply(lambda r: 3 if r['team_goals'] > r['opp_goals'] else (1 if r['team_goals'] == r['opp_goals'] else 0), axis=1)
matches['goal_diff'] = matches['team_goals'] - matches['opp_goals']
matches['pl_cum_points'] = matches.groupby(['season', 'team'])['match_points'].cumsum()
matches['pl_cum_gd'] = matches.groupby(['season', 'team'])['goal_diff'].cumsum()
matches['rank_metric'] = matches['pl_cum_points'] + (matches['pl_cum_gd'] / 1000.0)
matches['league_position'] = matches.groupby(['season', 'GW'])['rank_metric'].rank(ascending=False, method='min')
table_stats = matches[['season', 'GW', 'team', 'opponent', 'league_position']].drop_duplicates()

# Opponent attack/defence points
position_totals = all_players.groupby(['season', 'GW', 'team', 'position'])['total_points'].sum().reset_index()
attack_points = position_totals[position_totals['position'].isin(['FWD', 'MID'])].groupby(['season', 'GW', 'team'])['total_points'].sum().reset_index()
attack_points.rename(columns={'total_points': 'opp_att_points'}, inplace=True)
defence_points = position_totals[position_totals['position'].isin(['DEF', 'GK'])].groupby(['season', 'GW', 'team'])['total_points'].sum().reset_index()
defence_points.rename(columns={'total_points': 'opp_def_points'}, inplace=True)

# Merge all features
for pos in positions:
    df = raw_dfs[pos].copy()
    df = df.merge(table_stats[['season', 'GW', 'team', 'league_position']], on=['season', 'GW', 'team'], how='left')
    opp_pos = table_stats[['season', 'GW', 'team', 'league_position']].rename(columns={'team': 'opponent', 'league_position': 'opponent_league_position'})
    df = df.merge(opp_pos, on=['season', 'GW', 'opponent'], how='left')
    team_fpl = df.groupby(['season', 'GW', 'team'])['total_points'].sum().reset_index()
    team_fpl.rename(columns={'total_points': 'team_total_points'}, inplace=True)
    df = df.merge(team_fpl, on=['season', 'GW', 'team'], how='left')
    df = df.merge(attack_points.rename(columns={'team': 'opponent'}), on=['season', 'GW', 'opponent'], how='left')
    df = df.merge(defence_points.rename(columns={'team': 'opponent'}), on=['season', 'GW', 'opponent'], how='left')

    for col in ['league_position', 'opponent_league_position', 'team_total_points', 'opp_att_points', 'opp_def_points']:
        df[col] = df[col].fillna(10 if 'position' in col else 0)

    df['league_pos_trend'] = df.groupby(['season', 'team'])['league_position'].transform(lambda x: x.diff(3).fillna(0))
    df['opponent_position_trend'] = df.groupby(['season', 'opponent'])['opponent_league_position'].transform(lambda x: x.diff(3).fillna(0))
    df = df.drop_duplicates(subset=['name', 'season', 'GW'])
    out_file = os.path.join(FINAL_DATA_DIR, f'{pos}_all_seasons_final.csv')
    df.to_csv(out_file, index=False)
    print(f"  ✅ Saved {pos}: {len(df):,} rows → {out_file}")

print("\n✅ Feature engineering complete.")

# ============================================================
# PART 3: MODEL TRAINING & GRID SEARCH
# ============================================================
print("\n" + "="*70)
print("STEP 3: LSTM GRID SEARCH")
print("="*70)

# Load baseline and enhanced data
baseline_data = {}
enhanced_data = {}
for pos in positions:
    baseline_data[pos] = pd.read_csv(os.path.join(DATA_DIR, f'{pos}_all_seasons.csv'))
    enhanced_data[pos] = pd.read_csv(os.path.join(FINAL_DATA_DIR, f'{pos}_all_seasons_final.csv'))

BASELINE_FEATURES = [
    'total_points', 'goals_scored', 'assists',
    'expected_goals', 'expected_assists', 'minutes',
    'engineered_form', 'was_home', 'clean_sheets'
]
ENHANCED_FEATURES = BASELINE_FEATURES + [
    'league_position', 'opponent_league_position', 'team_total_points',
    'opp_att_points', 'opp_def_points', 'league_pos_trend', 'opponent_position_trend'
]

train_seasons = ['23_24']
val_seasons = ['24_25']
test_seasons = ['25_26']
train_seasons_enh = ['2023-24']
val_seasons_enh = ['2024-25']
test_seasons_enh = ['2025-26']
window_sizes = list(range(1, 16))

def create_sequences(df, window_sizes, features, train_s, val_s, test_s):
    results = {w: {'X_train': [], 'X_val': [], 'X_test': [], 'y_train': [], 'y_val': [], 'y_test': []} for w in window_sizes}
    df = df.copy()
    for col in features:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    for player in df['name'].unique():
        p_df = df[df['name'] == player].sort_values(['season', 'GW'])
        if len(p_df) < max(window_sizes) + 1:
            continue

        player_features = p_df[features].values
        constant_cols = np.where(np.std(player_features, axis=0) == 0)[0]
        scaler = MinMaxScaler()
        p_scaled = p_df.copy()
        if len(constant_cols) > 0:
            non_const_cols = [i for i in range(len(features)) if i not in constant_cols]
            if non_const_cols:
                scaled_vals = scaler.fit_transform(player_features[:, non_const_cols])
                full_scaled = np.zeros_like(player_features, dtype=np.float32)
                full_scaled[:, non_const_cols] = scaled_vals
            else:
                full_scaled = np.zeros_like(player_features, dtype=np.float32)
            p_scaled[features] = full_scaled
        else:
            p_scaled[features] = scaler.fit_transform(player_features)
        p_scaled[features] = np.nan_to_num(p_scaled[features].values, nan=0.0)

        for w in window_sizes:
            for i in range(len(p_scaled) - w):
                X_seq = p_scaled.iloc[i:i+w][features].values.astype(np.float32)
                y_val = p_scaled.iloc[i+w]['Next_GW_Points']
                if np.isnan(y_val) or np.isinf(y_val):
                    y_val = 0.0
                else:
                    y_val = float(y_val)
                season = p_scaled.iloc[i+w]['season']
                if season in train_s:
                    results[w]['X_train'].append(X_seq)
                    results[w]['y_train'].append(y_val)
                elif season in val_s:
                    results[w]['X_val'].append(X_seq)
                    results[w]['y_val'].append(y_val)
                elif season in test_s:
                    results[w]['X_test'].append(X_seq)
                    results[w]['y_test'].append(y_val)

    for w in window_sizes:
        for split in ['X_train', 'X_val', 'X_test']:
            arr = results[w][split]
            results[w][split] = np.array(arr, dtype=np.float32) if arr else np.empty((0, w, len(features)), dtype=np.float32)
        for split in ['y_train', 'y_val', 'y_test']:
            arr = results[w][split]
            results[w][split] = np.array(arr, dtype=np.float32) if arr else np.empty((0,), dtype=np.float32)
    return results

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='tanh', input_shape=input_shape),
        LSTM(32, return_sequences=False, activation='tanh'),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Baseline grid search
baseline_results = {}
for pos in positions:
    print(f"\n🔍 Baseline Grid Search: {pos}")
    seq = create_sequences(baseline_data[pos], window_sizes, BASELINE_FEATURES, train_seasons, val_seasons, test_seasons)
    pos_res = {}
    for w in window_sizes:
        X_train, y_train = seq[w]['X_train'], seq[w]['y_train']
        X_test, y_test = seq[w]['X_test'], seq[w]['y_test']
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        model = build_lstm((w, len(BASELINE_FEATURES)))
        model.fit(X_train, y_train, validation_data=(seq[w]['X_val'], seq[w]['y_val']),
                  epochs=50, batch_size=32, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
        _, mae = model.evaluate(X_test, y_test, verbose=0)
        pos_res[w] = mae
        print(f"  w={w}: MAE={mae:.3f}")
    baseline_results[pos] = pos_res

# Enhanced grid search
enhanced_results = {}
for pos in positions:
    print(f"\n🔍 Enhanced Grid Search: {pos}")
    seq = create_sequences(enhanced_data[pos], window_sizes, ENHANCED_FEATURES, train_seasons_enh, val_seasons_enh, test_seasons_enh)
    pos_res = {}
    for w in window_sizes:
        X_train, y_train = seq[w]['X_train'], seq[w]['y_train']
        X_test, y_test = seq[w]['X_test'], seq[w]['y_test']
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        model = build_lstm((w, len(ENHANCED_FEATURES)))
        model.fit(X_train, y_train, validation_data=(seq[w]['X_val'], seq[w]['y_val']),
                  epochs=50, batch_size=32, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
        _, mae = model.evaluate(X_test, y_test, verbose=0)
        pos_res[w] = mae
        print(f"  w={w}: MAE={mae:.3f}")
    enhanced_results[pos] = pos_res

# Save grid search results
baseline_df = pd.DataFrame(baseline_results).T
enhanced_df = pd.DataFrame(enhanced_results).T
baseline_df.to_csv(os.path.join(RESULTS_DIR, 'baseline_mae_results.csv'))
enhanced_df.to_csv(os.path.join(RESULTS_DIR, 'enhanced_mae_results.csv'))

# ============================================================
# PART 4: ABLATION STUDY
# ============================================================
print("\n" + "="*70)
print("STEP 4: ABLATION STUDY")
print("="*70)

ablation_groups = {
    'Full Model': BASELINE_FEATURES,
    'No Personal Form': [f for f in BASELINE_FEATURES if f not in ['total_points', 'goals_scored', 'assists', 'engineered_form']],
    'No Expected Stats': [f for f in BASELINE_FEATURES if f not in ['expected_goals', 'expected_assists']],
    'No Match Context': [f for f in BASELINE_FEATURES if f not in ['was_home', 'minutes', 'clean_sheets']]
}
best_windows = {'GK': 7, 'DEF': 4, 'MID': 10, 'FWD': 9}  # from baseline grid search (will be overridden by actual optimal)

# Re-determine optimal windows from baseline results
baseline_df = pd.DataFrame(baseline_results).T
best_windows = baseline_df.idxmin(axis=1).to_dict()

ablation_results = []
for pos in positions:
    w = best_windows[pos]
    print(f"\nPosition: {pos} (window={w})")
    for group_name, features in ablation_groups.items():
        seq = create_sequences(baseline_data[pos], [w], features, train_seasons, val_seasons, test_seasons)
        X_train, y_train = seq[w]['X_train'], seq[w]['y_train']
        X_test, y_test = seq[w]['X_test'], seq[w]['y_test']
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        model = build_lstm((w, len(features)))
        model.fit(X_train, y_train, validation_data=(seq[w]['X_val'], seq[w]['y_val']),
                  epochs=50, batch_size=32, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
        _, mae = model.evaluate(X_test, y_test, verbose=0)
        ablation_results.append({'Position': pos, 'Group': group_name, 'Features': len(features), 'MAE': mae})
        print(f"  {group_name}: MAE={mae:.3f}")

df_ablation = pd.DataFrame(ablation_results)
df_ablation.to_csv(os.path.join(RESULTS_DIR, 'ablation_results.csv'), index=False)

# ============================================================
# PART 5: COMPARISON PLOTS
# ============================================================
print("\n" + "="*70)
print("STEP 5: GENERATING PLOTS")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for pos in positions:
    axes[0].plot(baseline_df.columns, baseline_df.loc[pos], marker='o', label=pos)
axes[0].set_title('Baseline (9 features)')
axes[0].set_xlabel('Window Length'); axes[0].set_ylabel('Test MAE')
axes[0].legend(); axes[0].grid(alpha=0.3)

enhanced_df = pd.DataFrame(enhanced_results).T
for pos in positions:
    axes[1].plot(enhanced_df.columns, enhanced_df.loc[pos], marker='o', label=pos)
axes[1].set_title('Enhanced (16 features)')
axes[1].set_xlabel('Window Length'); axes[1].set_ylabel('Test MAE')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comparison_plots.png'), dpi=150)
plt.close()

# Ablation bar plot
pivot = df_ablation.pivot(index='Position', columns='Group', values='MAE')
pivot = pivot[['Full Model', 'No Personal Form', 'No Expected Stats', 'No Match Context']]
pivot.plot(kind='bar', figsize=(10,6))
plt.title('Ablation Study: MAE by Feature Group')
plt.ylabel('Test MAE')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'ablation_plot.png'), dpi=150)
plt.close()

print(f"\n✅ All results saved to '{RESULTS_DIR}'")
print("Pipeline completed successfully.")

