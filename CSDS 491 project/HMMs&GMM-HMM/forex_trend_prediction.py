import os
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import re
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define command-line arguments
parser = argparse.ArgumentParser(description='Forex Trend Prediction using HMM and GMM-HMM models')
parser.add_argument('-c', '--currency_pair', type=str, default='USD_CNY',
                    help='Currency pair to predict (format: XXX_YYY)')
parser.add_argument('-m', '--model', type=str, default='GaussianHMM',
                    choices=['GaussianHMM', 'GMMHMM'],
                    help='Model to use for prediction')

args = parser.parse_args()

# Validate the currency pair format
if not re.match(r'^[A-Z]{3}_[A-Z]{3}$', args.currency_pair):
    print("Invalid currency pair format. Please use the format: XXX_YYY")
    exit(1)

currency_pair = args.currency_pair
model_type = args.model

class ForexData:
    def __init__(self, currency_pair, test_size=0.1):
        self.currency_pair = currency_pair
        self.train_data, self.test_data = self._load_and_split_data(test_size)

    def _load_and_split_data(self, test_size):
        data = pd.read_csv(f"{currency_pair}_5_years.csv")
        data = data.iloc[::-1]
        train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
        return train_data, test_data
        
    @property
    def test_data_features(self):
        return ForexPredictor._extract_features(self.test_data)

class ForexPredictor(object):
    def __init__(self, currency_pair, model_type, forex_data, n_hidden_states=4, n_latency_days=10,
                 n_steps_frac_change=50, n_steps_frac_high=10, n_steps_frac_low=10):
        self._init_logger()
        self.currency_pair = currency_pair
        self.n_latency_days = n_latency_days
        self.forex_data = forex_data

        if model_type == 'GaussianHMM':
            self.hmm = GaussianHMM(n_components=n_hidden_states)
        elif model_type == 'GMMHMM':
            self.hmm = GMMHMM(n_components=n_hidden_states, n_mix=4)
        else:
            raise ValueError("Invalid model type. Use 'GaussianHMM' or 'GMMHMM'")

        self._compute_all_possible_outcomes(n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)

    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def fit(self):
        self._logger.info('>>> Extracting Features')
        feature_vector = ForexPredictor._extract_features(self.forex_data.train_data)
        self._logger.info('Features extraction Completed <<<')

        self.hmm.fit(feature_vector)

    def _compute_all_possible_outcomes(self, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low):
        frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_steps_frac_low)
        self._possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range)))

    @staticmethod
    def _extract_features(data):
        open_price = np.array(data['open'])
        close_price = np.array(data['close'])
        high_price = np.array(data['high'])
        low_price = np.array(data['low'])

        frac_change = (close_price - open_price) / open_price
        frac_high = (high_price - open_price) / open_price
        frac_low = (open_price - low_price) / open_price

        return np.column_stack((frac_change, frac_high, frac_low))

    def predict_close_price(self, day_index):
        open_price = self.forex_data.test_data.iloc[day_index]['open']
        predicted_frac_change, _, _ = self._get_most_probable_outcome(day_index)
        return open_price * (1 + predicted_frac_change)

    def _get_most_probable_outcome(self, day_index):
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self.forex_data.test_data.iloc[previous_data_start_index: previous_data_end_index]
        previous_data_features = ForexPredictor._extract_features(previous_data)

        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack((previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(outcome_score)]

        return most_probable_outcome

    def predict_close_prices_for_days(self, days, with_plot=False):
        predicted_close_prices = []
        for day_index in tqdm(range(days)):
            predicted_close_prices.append(self.predict_close_price(day_index))

        if with_plot:
            test_data = self.forex_data.test_data[0: days]
            test_data['date'] = pd.to_datetime(test_data['date'])
            days = test_data['date'].values
            actual_close_prices = test_data['close']
            fig = plt.figure(figsize=(12, 6))

            axes = fig.add_subplot(111)
            axes.plot(days, actual_close_prices, 'b-', label="actual")
            axes.plot(days, predicted_close_prices, 'r-', label="predicted")
            axes.set_title(f'{self.currency_pair} - {model_type}')
            axes.set_xlabel('Date')
            axes.set_ylabel('Close Price')
            axes.grid(True)

            fig.autofmt_xdate()

            plt.legend()
            plt.savefig(f'{self.currency_pair}_{model_type}_plot.png', dpi=300)
            plt.show()

#        # Save the plotting data to a CSV file
#        plotting_data = pd.DataFrame({
#            'date': days,
#            'actual_close_price': actual_close_prices,
#            'predicted_close_price': predicted_close_prices
#        })
#        plotting_data.to_csv(f'{self.currency_pair}_{model_type}_plot_data.csv', index=False)

        return predicted_close_prices

forex_data = ForexData(currency_pair=currency_pair)
forex_predictor = ForexPredictor(currency_pair=currency_pair, model_type=model_type, forex_data=forex_data)
forex_predictor.fit()
num_days = len(forex_data.test_data)
#forex_predictor.predict_close_prices_for_days(num_days, with_plot=True)
#
#actual_close_prices = forex_data.test_data['close'].values
#predicted_close_prices = np.array(forex_predictor.predict_close_prices_for_days(num_days))
actual_close_prices = forex_data.test_data['close'].values
predicted_close_prices = np.array(forex_predictor.predict_close_prices_for_days(num_days, with_plot=True))

# Save the plotting data to a CSV file
plotting_data = pd.DataFrame({
    'date': forex_data.test_data['date'].values[:num_days],
    'actual_close_price': actual_close_prices,
    'predicted_close_price': predicted_close_prices
})
plotting_data.to_csv(f'{currency_pair}_{model_type}_plot_data.csv', index=False)

mae = mean_absolute_error(actual_close_prices, predicted_close_prices)
mse = mean_squared_error(actual_close_prices, predicted_close_prices)
rmse = np.sqrt(mse)
r2 = r2_score(actual_close_prices, predicted_close_prices)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

if args.model == "GMMHMM":
    from scipy.stats import norm

    feature_data = forex_data.test_data_features
    min_range = np.min(feature_data[:, 0])
    max_range = np.max(feature_data[:, 0])
    x_range = np.linspace(min_range, max_range, 1000)
    
    gmm_probs = np.zeros((forex_predictor.hmm.n_components, len(x_range)))

    for i in range(forex_predictor.hmm.n_components):
        for j in range(forex_predictor.hmm.n_mix):
            gmm_probs[i] += forex_predictor.hmm.weights_[i][j] * norm.pdf(
                x_range,
                forex_predictor.hmm.means_[i][j][0],
                np.sqrt(forex_predictor.hmm.covars_[i][j][0])
            )

    plt.figure(figsize=(10, 6))

    for i in range(forex_predictor.hmm.n_components):
        plt.plot(x_range, gmm_probs[i], label=f"State {i + 1}")

    # Add GMM curve
    gmm_curve = np.sum(gmm_probs, axis=0)
    plt.plot(x_range, gmm_curve, label="GMM", linestyle="--", color="black")

    plt.title("GMMHMM Probability Distribution Functions")
    plt.xlabel("Fraction Change")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig("gmmhmm_probability_distributions.png")
    plt.show()



