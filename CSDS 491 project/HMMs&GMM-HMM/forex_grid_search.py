import numpy as np
from sklearn.metrics import mean_absolute_error
from forex_trend_prediction import ForexData
from forex_trend_prediction import ForexPredictor

def evaluate_model(model_type, currency_pair, n_hidden_states, n_latency_days):
    forex_data = ForexData(currency_pair=currency_pair)
    forex_predictor = ForexPredictor(currency_pair=currency_pair, model_type=model_type, forex_data=forex_data,
                                     n_hidden_states=n_hidden_states, n_latency_days=n_latency_days)
    forex_predictor.fit()
    num_days = len(forex_data.test_data)
    predicted_close_prices = np.array(forex_predictor.predict_close_prices_for_days(num_days))
    actual_close_prices = forex_data.test_data['close'].values
    mae = mean_absolute_error(actual_close_prices, predicted_close_prices)
    return mae

def grid_search(model_type, currency_pair, hidden_states_range, latency_days_range):
    best_params = None
    best_mae = float('inf')

    for n_hidden_states in hidden_states_range:
        for n_latency_days in latency_days_range:
            mae = evaluate_model(model_type, currency_pair, n_hidden_states, n_latency_days)
            print(f"Model: {model_type}, Hidden States: {n_hidden_states}, Latency Days: {n_latency_days}, MAE: {mae}")

            if mae < best_mae:
                best_mae = mae
                best_params = {'n_hidden_states': n_hidden_states, 'n_latency_days': n_latency_days}

    return best_params, best_mae

if __name__ == "__main__":
    currency_pair = 'USD_CNY'
    hidden_states_range = range(2, 10)
    latency_days_range = range(5, 20)

    best_params_gaussian_hmm, best_mae_gaussian_hmm = grid_search('GaussianHMM', currency_pair, hidden_states_range, latency_days_range)
    print(f"Gaussian HMM - Best Parameters: {best_params_gaussian_hmm}, Best MAE: {best_mae_gaussian_hmm}")

    best_params_gmmhmm, best_mae_gmmhmm = grid_search('GMMHMM', currency_pair, hidden_states_range, latency_days_range)
    print(f"GMMHMM - Best Parameters: {best_params_gmmhmm}, Best MAE: {best_mae_gmmhmm}")

