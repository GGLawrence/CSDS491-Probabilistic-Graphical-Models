# CSDS491 Probabilistic Graphical Models
 CSDS491 Probabilistic Graphical Models

This is a Final porject file.

### Team member:
Sinan He, Qingzhe Guo, Ziming Cui in CWRU

### Predicting Forex Trends: A Comparative Study of Probabilistic Graphical Models and Beyond
Project Proposal
Sinan He, Qingzhe Guo, Ziming Cui

#### Explanation and motivation
Foreign exchange (Forex) is the global market for trading currencies, which operates 24
hours a day, 5 days a week. Predicting Forex trends is crucial for traders, financial
institutions, and policy-makers, as it can help optimize trading strategies, manage risk,
and facilitate decision-making processes. As international students, we are particularly
sensitive to fluctuations in exchange rates, which directly impact our finances and the
affordability of education and living expenses in our daily life.

The goal of this project is to investigate the use of Probabilistic Graphical Models
(PGMs) for predicting Forex trends and compare their performance with other popular
machine learning models, such as Long Short-Term Memory (LSTM) and eXtreme
Gradient Boosting (XGBoost). The models to be explored in this project include Hidden
Markov Models (HMMs), Gaussian Mixture Model-Hidden Markov Models (GMMHMMs),
and Markov Random Fields (MRFs). Success in this project means developing
accurate and reliable models for predicting short-term Forex trends, which can be
evaluated through performance metrics such as Mean Absolute Error (MAE), Root
Mean Squared Error (RMSE), and R-squared (R²) values.

#### Approach and rationale
We chose to focus on Probabilistic Graphical Models due to their ability to provide a
compact representation of complex probability distributions and facilitate efficient
inference and learning. HMMs, GMM-HMMs, and maybe other potential appropriate
PGMs are selected for their diverse ways of representing and reasoning about the
complex relationships in the Forex market.

To simplify the problem and limit the scope of the project, we will:

Focus on predicting short-term trends (e.g., intraday or daily) instead of long-term
trends.

Concentrate on major currency pairs, such as CNY/USD, EUR/USD, and USD/JPY, to
reduce computational complexity and enable a more in-depth analysis.
Use historical price data as the primary input, without incorporating external factors
such as macroeconomic indicators or news sentiment.
The project will be divided into the following stages:
Data collection and preprocessing: Collect historical price data for selected currency
pairs, clean and preprocess the data, and create training and testing datasets.
Model development: Implement HMMs, GMM-HMMs, LSTM, XGBoost and other
potential models using appropriate libraries and frameworks, and train them on the
prepared datasets.
Model evaluation: Evaluate the performance of the models using the testing dataset and
choosing appropriate performance metrics, such as MAE, RMSE, and R² values.
Comparison and analysis: Compare the performance of the different models and
analyze their strengths and weaknesses in the context of Forex trend prediction.
By exploring the capabilities of Probabilistic Graphical Models and comparing them with
other popular machine learning models, we aim to identify the most suitable approach
for predicting Forex trends. This project will contribute to the development of more
accurate and reliable models for Forex trend prediction, ultimately benefiting various
stakeholders in their decision-making processes.

#### Project Division
In our group of three members, we have initially divided the roles as follows, with the
understanding that adjustments may be made based on project needs as we progress:
1. Qingzhe Guo will lead Data Collection, Preprocessing, and Project Management. He
will prepare the training and testing datasets and ensure efficient collaboration among
team members. He will also contribute to XGBoost model.
2. Sinan He will be responsible for implementing and fine-tuning PGMs, such as HMMs
and GMM-HMMs, and other potential PGMs. He will also manage the project research
and write-up.
3. Ziming Cui will focus on implementing and optimizing LSTM models, and participating
in data preprocessing, model evaluation and analysis, testing each model and feedback
to every teammate.
This division of roles allows each group member to contribute their expertise and
collaborate effectively, ensuring the project's success.
