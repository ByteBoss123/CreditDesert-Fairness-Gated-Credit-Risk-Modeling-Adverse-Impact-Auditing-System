"""Consumer credit training with untouched final test split."""
import argparse
import pickle
import pandas as pd
from paths import DATA, OUTPUTS
from features_consumer import build_features, build_target
from training import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    source = DATA / 'consumer_credit_data.csv'
    df = pd.read_csv(source)
    result = train_model(build_features(df), build_target(df), 'CreditDesert_Consumer', source, args.register)
    with open(OUTPUTS / 'consumer_model_results.pkl', 'wb') as f:
        pickle.dump(result, f)
