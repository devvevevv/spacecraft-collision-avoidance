import pandas as pd

RAW_DATA_PATH = r"../data/raw/train_data.csv"
OUTPUT_PATH = r"../data/processed/preprocessed.csv"
RISK_THRESHOLD = -6
MIN_VALID_RISK = -30

SELECTED_FEATURES = [
    'time_to_tca',
    'max_risk_estimate',
    'max_risk_scaling',
    'mahalanobis_distance',
    'miss_distance',
    'c_position_covariance_det',
    'c_obs_used'
]

def classify_event(r_minus2, r):
    """
    :param r_minus2: float, risk from the latest CDM where time_to_tca >= 2
    :param r: float, risk from the CDM closest to TCA (time_to_tca < 2)
    :return: str, one of ['anomalous', 'non-anomalous', 'ignore']
    """

    if r_minus2 < RISK_THRESHOLD <= r:
        return "anomalous"
    elif (r_minus2 < RISK_THRESHOLD and r < RISK_THRESHOLD) or (r_minus2 >= RISK_THRESHOLD and r >= RISK_THRESHOLD):
        return "non-anomalous"
    else:
        return "ignore"  # high to low risk – ignored

df = pd.read_csv(RAW_DATA_PATH)
df = df[df['risk'] > MIN_VALID_RISK]

r_df = df[df['time_to_tca'] < 2]
r_df = r_df.sort_values(['event_id', 'time_to_tca'])
r_final = r_df.groupby('event_id').first().reset_index()[['event_id', 'risk']]
r_final = r_final.rename(columns={'risk': 'r_final'})

r2_df = df[df['time_to_tca'] >= 2]
r2_df = r2_df.sort_values(['event_id', 'time_to_tca'], ascending=[True, False])
r2_latest = r2_df.groupby('event_id').first().reset_index()
r2_latest = r2_latest[['event_id', 'risk'] + SELECTED_FEATURES]
r2_latest = r2_latest.rename(columns={'risk': 'r_minus2'})

labeled_df = pd.merge(r2_latest, r_final, on='event_id', how='inner')
labeled_df['label'] = labeled_df.apply(lambda row: classify_event(row['r_minus2'], row['r_final']), axis=1)

labeled_df = labeled_df[labeled_df['label'] != 'ignore']

agg_stats = r2_df.groupby('event_id')['risk'].agg(
    number_CDMs='count',
    mean_risk_CDMs='mean',
    std_risk_CDMs='std'
).reset_index()

final_df = pd.merge(labeled_df, agg_stats, on='event_id', how='left')
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved preprocessed data to {OUTPUT_PATH}")
