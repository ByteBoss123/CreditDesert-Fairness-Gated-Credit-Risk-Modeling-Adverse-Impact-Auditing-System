"""Train HMDA approval/denial models; this target is NOT repayment default."""
import argparse
import pickle
import pandas as pd
from paths import DATA, OUTPUTS
from features import build_features, build_demographics
from training import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true', help='Register candidate versions, without production promotion')
    args = parser.parse_args()
    source = DATA / 'hmda_va_2024_modeling.parquet'
    df = pd.read_parquet(source)
    X, y, demo = build_features(df), df['approved'], build_demographics(df)
    results = {}
    for key, name, features in [('A','CreditDesert_HMDA_A',X), ('B','CreditDesert_HMDA_B',X.drop(columns=['tract_minority_population_percent']))]:
        result = train_model(features, y, name, source, args.register)
        result['demo_test'] = demo.iloc[result['test_indices']]
        results[key] = result
    with open(OUTPUTS / 'model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
