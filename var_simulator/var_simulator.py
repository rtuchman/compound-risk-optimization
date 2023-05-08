import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union
from concurrent.futures import ThreadPoolExecutor

import arch
from arch import univariate

CPU_COUNT = os.cpu_count()


class VarSimulator:

    def __init__(self, df_prices: pd.DataFrame):
        self.df_prices = df_prices
        self.returns = df_prices.pct_change().dropna()
        self.log_returns = np.log(1 + self.returns)
        self.assets = self.log_returns.columns

    def calculate_volatility(self, p: int, q: int, n_tests: int) -> Union[Dict, Dict, Dict, Dict]:
        models = {}
        forecasts = {}
        historical_volatility = {}
        garch_volatility = {}
        for asset in self.assets:
            model = arch.arch_model(self.log_returns[asset], p=p, q=q, vol="Garch", dist="Normal")
            models[asset] = model.fit(disp=False)
            forecasts[asset] = models[asset].forecast(horizon=n_tests, method='simulation', reindex=False)
            historical_volatility[asset] = self.log_returns[asset].rolling(window=n_tests - 1).std().iloc[-1]
            garch_volatility[asset] = forecasts[asset].variance.iloc[-1]

        return models, forecasts, historical_volatility, garch_volatility

    def generate_random_price_trajectories(self, forecasts: univariate.base.ARCHModelForecast,
                                           n_periods: int = 720) -> Dict:
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
        user_debt = (df_top_borrow_positions['totalUnderlyingBorrowed'] - df_top_borrow_positions[
            'totalUnderlyingRepaid']) - (df_top_borrow_positions['totalUnderlyingSupplied'] - df_top_borrow_positions[
            'totalUnderlyingRedeemed'])
        return user_debt.apply(lambda x: max(0, x))

    def _calculate_single_var_measurement(self, df_top_borrow_positions: pd.DataFrame,
                                          forecasts: univariate.base.ARCHModelForecast, simulation_index: int):
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

    def calculate_var(self, df_top_borrow_positions: pd.DataFrame, forecasts: univariate.base.ARCHModelForecast,
                      n_simulations: int):
        self.var_results = {}
        df_top_borrow_positions = df_top_borrow_positions[
            df_top_borrow_positions['symbol'].isin(list(self.df_prices.columns))].reset_index(drop=True)
        df_top_borrow_positions['userDebt'] = var_simulator._calc_user_debt(df_top_borrow_positions)

        futures = []
        with ThreadPoolExecutor(CPU_COUNT * 2) as executor:
            for i in range(n_simulations):
                futures.append(
                    executor.submit(self._calculate_single_var_measurement, df_top_borrow_positions, forecasts, i))

            for future in tqdm(futures):
                res = future.result()
                self.var_results[res['sim_name']] = res['sim_result']

    def plot_price_trajectories(self, asset: str, n_simulations: int):
        plt.figure(figsize=(10, 5))
        plt.plot(np.vstack(
            var_simulator.generate_random_price_trajectories(forecasts)[asset] for _ in range(n_simulations)).T)
        plt.title(f"Random Price Trajectories for {asset}")
        plt.xlabel('Time (hours)')
        plt.ylabel('Price (USD)')
        plt.show()

    def plot_var(bins: int = 30):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=[v['var_value'] for _, v in self.var_results.items()], bins=bins, ax=ax)
        plt.title('Distribution of VaR Estimates')
        plt.xlabel('Value at Risk (USD)')
        plt.ylabel('Frequency')
        plt.show()