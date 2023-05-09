import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import arch
from arch import univariate

CPU_COUNT = os.cpu_count()


class VarSimulator:
    """
    A class that can simulate price trajectories for given assets and calculate Value at Risk (VaR) estimates using
    historical data and Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model. VaR is the maximum loss
    that an investor can incur, within a certain confidence interval, over a given period of time.
    """
    def __init__(self, df_prices: pd.DataFrame):
        """
         Initializes the VarSimulator object with a DataFrame of historical asset prices.

         :param df_prices: A pandas DataFrame containing the historical prices of the assets to be simulated.

         This method initializes the VarSimulator object by calculating the logarithmic returns of the assets in the
         given DataFrame, and creating a list of asset names. It also initializes four dictionaries to store the GARCH
         models, forecasts, historical volatilities, and GARCH volatilities for each asset.
         """
        self.df_prices = df_prices
        self.returns = df_prices.pct_change().dropna()
        self.log_returns = np.log(1 + self.returns)
        self.assets = self.log_returns.columns
        self.models = {}
        self.forecasts = {}
        self.historical_volatility = {}
        self.garch_volatility = {}

    def calculate_volatility(self, p: int, q: int, n_tests: int) -> Union[Dict, Dict, Dict, Dict]:
        """
        Calculates the GARCH volatility forecasts and historical volatilities for each asset in the dataset.

        :param p: The order of the autoregressive (AR) term.
        :param q: The order of the moving average (MA) term.
        :param n_tests: The number of periods to be used in the rolling standard deviation calculation.

        :return: A dictionary containing the GARCH models, forecasts, historical volatilities, and GARCH volatilities
        for each asset in the dataset.
        """
        for asset in self.assets:
            model = arch.arch_model(self.log_returns[asset], p=p, q=q, vol="Garch", dist="Normal", mean='Zero', rescale=False)
            self.models[asset] = model.fit(disp=False)
            self.forecasts[asset] = self.models[asset].forecast(horizon=n_tests, method='simulation', reindex=False)
            self.historical_volatility[asset] = self.log_returns[asset].rolling(window=n_tests - 1).std().iloc[-1]
            self.garch_volatility[asset] = self.forecasts[asset].variance.iloc[-1]

    def generate_random_price_trajectories(self, forecasts: univariate.base.ARCHModelForecast,
                                           n_periods: int = 720) -> Dict:
        """
        Generates random price trajectories for given assets for a given number of periods, based on the GARCH forecasts.

        :param forecasts: The forecast objects obtained from the GARCH model.
        :param n_periods: The number of periods to be simulated.

        :return: A dictionary containing the price trajectories for the given assets.
        """
        self.n_periods = n_periods
        price_trajectories = {}
        for asset in self.assets:
            S0 = self.df_prices[asset].values[-1]
            dt = 1 / 24

            sigma = np.sqrt(forecasts[asset].variance.values.mean())
            r = self.log_returns.mean()[asset]
            Z = np.random.normal(size=(n_periods))

            tmp_prices = np.zeros((n_periods + 1))
            tmp_prices[0] = S0
            for i in range(n_periods):
                drift = (r - 0.5 * sigma ** 2) * dt
                diffusion = sigma * np.sqrt(dt) * Z[i]
                tmp_prices[i + 1] = np.exp(np.log(tmp_prices[i]) + drift + diffusion)

            price_trajectories[asset] = tmp_prices

        return price_trajectories

    def _calc_user_debt(self, df_top_borrow_positions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the user debt for each account based on the top borrow positions.

        :param df_top_borrow_positions: A pandas DataFrame containing the top borrow positions for the accounts.

        :return: A pandas DataFrame containing the user debt for each account.
        """
        user_debt = (df_top_borrow_positions['totalUnderlyingBorrowed'] - df_top_borrow_positions[
            'totalUnderlyingRepaid']) - (df_top_borrow_positions['totalUnderlyingSupplied'] - df_top_borrow_positions[
            'totalUnderlyingRedeemed'])
        return user_debt.apply(lambda x: max(0, x))

    def _calculate_single_var_measurement(self, df_top_borrow_positions: pd.DataFrame,
                                          forecasts: univariate.base.ARCHModelForecast, simulation_index: int):
        """
        Calculates the user debt for each account based on the top borrow positions.

        :param df_top_borrow_positions: A pandas DataFrame containing the top borrow positions for the accounts.

        :return: A pandas DataFrame containing the user debt for each account.
        """
        price_trajectories = self.generate_random_price_trajectories(forecasts)
        hourly_user_debts = []
        for acount_id, df_account in df_top_borrow_positions.groupby('account_id'):
            user_assets = list(df_account['symbol'].unique())
            user_debt = np.maximum(0, np.sum(
                df_account[df_account['symbol'] == s]['userDebt'].values[0] * price_trajectories[s] for s in
                user_assets))
            hourly_user_debts.append(user_debt)
        bad_debt = np.sum(hourly_user_debts, axis=0)
        var_prob = np.percentile(bad_debt, 5)
        return {'sim_name': f"simulation_{simulation_index}",
                'sim_result': {"bad_debt": bad_debt, "var_value": var_prob}}

    def calculate_var(self, df_top_borrow_positions: pd.DataFrame, n_simulations: int):
        """
        Calculates the VaR for given top borrow positions and the GARCH forecasts for a given number of simulations.

        :param df_top_borrow_positions: A pandas DataFrame containing the top borrow positions for the accounts.
        :param n_simulations: The number of simulations to be run.

        :return: None
        """
        self.var_results = {}
        df_top_borrow_positions = df_top_borrow_positions[
            df_top_borrow_positions['symbol'].isin(list(self.df_prices.columns))].reset_index(drop=True)
        df_top_borrow_positions['userDebt'] = self._calc_user_debt(df_top_borrow_positions)

        futures = []
        with ThreadPoolExecutor(CPU_COUNT * 2) as executor:
            for i in range(n_simulations):
                futures.append(
                    executor.submit(self._calculate_single_var_measurement, df_top_borrow_positions, self.forecasts, i))

            for future in tqdm(futures):
                res = future.result()
                self.var_results[res['sim_name']] = res['sim_result']

    def plot_price_trajectories(self, asset: str, n_simulations: int):
        """
        Plots random price trajectories for a given asset based on GARCH model simulations.

        :param asset: A string indicating the name of the asset.
        :param n_simulations: An integer indicating the number of simulations to be run.

        This method generates a plot of n_simulations random price trajectories for the given asset based on GARCH model
        simulations. It uses the `generate_random_price_trajectories` method to generate the price trajectories and plots
        them using matplotlib.

        :return: None
        """
        plt.figure(figsize=(10, 5))
        plt.plot(np.vstack(
            self.generate_random_price_trajectories(self.forecasts)[asset] for _ in range(n_simulations)).T)
        plt.title(f"Random Price Trajectories for {asset}")
        plt.xlabel('Time (hours)')
        plt.ylabel('Price (USD)')
        plt.show()

    def plot_var(self, bins: int = 30):
        """
        Plots a histogram of the VaR estimates obtained from the simulations.

        :param bins: Number of bins for the histogram.

        This method plots a histogram of the VaR estimates obtained from the simulations. The `bins` parameter specifies the
        number of bins to be used for the histogram.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=[v['var_value'] for _, v in self.var_results.items()], bins=bins, ax=ax)
        plt.title('Distribution of VaR Estimates')
        plt.xlabel('Value at Risk (USD)')
        plt.ylabel('Frequency')
        plt.show()