import json
import os
import requests
import datetime
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from graphqlclient import GraphQLClient


class CompoundSubgraphQuery:
    """
    A class that provides methods to query the Compound subgraph using GraphQL.
    """
    def __init__(self):
        """
        Initializes a new instance of the CompoundSubgraphQuery class.
        """
        self.client = GraphQLClient('https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2')

    def gql_query(self, query_string: str, variables: Optional[Dict] = None) -> Dict:
        """
        Executes a GraphQL query against the Compound Subgraph.

        :param query_string: The query string to be executed.
        :param variables: (optional) A dictionary containing the variables used in the query.
        :return: A dictionary containing the query result.
        """
        return json.loads(self.client.execute(query_string, variables))

    def query_borrow_positions(self, query_string: str, last_id: str = '', pkl_index: int = 0):
        """
        Queries the Compound v2 subgraph to retrieve all borrow positions.
        The results are saved as pickled dataframes to allow for resuming the download in case of connection issues.

        :param query_string: A string representing the GraphQL query to execute.
        :param last_id: The ID of the last account retrieved. Used for pagination.
        :param pkl_index: The index number to use for the pickled file. Used for saving multiple files.
        """
        with tqdm() as pbar:
            while True:
                variables = {'last': last_id}
                result = self.gql_query(query_string, variables)
                accounts = result['data']['accounts']
                if not accounts:
                    break
                last_id = result['data']['accounts'][-1]['id']
                pkl_path = f"data/borrow_positions_{pkl_index}.pkl"
                pd.DataFrame(accounts).to_pickle(pkl_path)  
                pbar.set_description(f"saved file to: {pkl_path}")
                pkl_index+=1
                pbar.update(1)
                
    def query_top_compound_markets(self, query_string: str) -> pd.DataFrame:
        """
        Queries the Compound v2 subgraph for the top markets by total supply in USD, and returns the results as a pandas dataframe.

        :param query_string: The GraphQL query string used to fetch the top markets from the subgraph.
        :return: A pandas dataframe containing the top Compound markets, sorted by total supply in USD in descending order.
        """
        top_markets_json = self.gql_query(query_string)['data']['markets']
        df_top_markets = pd.DataFrame(top_markets_json)
        markets_numeric_columns = ['reserves', 'supplyRate', 'totalBorrows', 'totalSupply', 'underlyingPriceUSD']
        for col in markets_numeric_columns:
            df_top_markets[col] = df_top_markets[col].astype(np.float64)
        df_top_markets['totalSupplyUSD'] = df_top_markets.apply(lambda x: x['totalSupply']*x['underlyingPriceUSD'], axis=1)
        return df_top_markets.sort_values(by='totalSupplyUSD', ascending=False).reset_index(drop=True)

    @staticmethod
    def read_top_borrow_accounts(top_n: int = 1000) -> pd.DataFrame:
        """
        Read the pickled borrow accounts data from the `data` directory, concatenates them and returns the top n accounts
        sorted by totalBorrowValueInEth column in descending order.

        :param top_n: Number of top accounts to return (default 1000)
        :return: A pandas dataframe containing the concatenated data of top n accounts.
        """
        df_borrow_accounts = pd.DataFrame()
        pkl_files = [x for x in sorted(os.listdir('data')) if 'borrow_positions' in x]
        for pkl_file in pkl_files:
            df_borrow_accounts = pd.concat([df_borrow_accounts, pd.read_pickle(f"data/{pkl_file}")])
        return df_borrow_accounts.sort_values(by='totalBorrowValueInEth', ascending=False).head(top_n).reset_index(drop=True)

    @staticmethod
    def cast_numeric_cols_to_float(df: pd.DataFrame) -> pd.DataFrame:
        """
        Casts columns with numeric values to float64.

        :param df: DataFrame with columns to be casted.
        :return: DataFrame with columns casted to float64.
        """
        numeric_columns = ['totalBorrowValueInEth', 'totalCollateralValueInEth',
                            'cTokenBalance', 'totalUnderlyingSupplied',
                            'totalUnderlyingRedeemed', 'totalUnderlyingBorrowed',
                            'totalUnderlyingRepaid', 'storedBorrowBalance']
        for col in numeric_columns:
            df[col] = df[col].astype(np.float64)
        return df

    def parse_borrow_positions(self, df_accounts: pd.DataFrame) -> pd.DataFrame:
        """
         Parse the borrow positions dataframe to obtain a cleaned and standardized dataframe.

         :param df_accounts: The dataframe obtained from Compound's subgraph that contains borrow positions data.
         :type df_accounts: pd.DataFrame
         :return: A cleaned and standardized dataframe that contains parsed borrow positions data.
         :rtype: pd.DataFrame
         """
        borrow_tokens_list = []
        for tmp_account_id, df_account in df_accounts.groupby('id'):
            tmp_totalBorrowValueInEth, tmp_totalCollateralValueInEth = df_account['totalBorrowValueInEth'], df_account['totalCollateralValueInEth']
            for token in df_account['tokens'].values[0]:
                tmp_token_dict = {'account_id': tmp_account_id, 'totalBorrowValueInEth': tmp_totalBorrowValueInEth, 'totalCollateralValueInEth': tmp_totalCollateralValueInEth}
                tmp_token_dict.update(token)
                borrow_tokens_list.append(tmp_token_dict)

        df_top_borrow_positions = self.cast_numeric_cols_to_float(pd.DataFrame(borrow_tokens_list))
        df_top_borrow_positions = df_top_borrow_positions.rename(columns={'symbol': 'cToken_symbol'})
        df_top_borrow_positions['symbol'] = df_top_borrow_positions['cToken_symbol'].apply(lambda x: x[1:])

        return df_top_borrow_positions.sort_values(by='totalBorrowValueInEth', ascending=False).reset_index(drop=True)


def fetch_cg_token_ids(top_markets_symbols: List[str]) -> List[str]:
    """
    Fetches CoinGecko token IDs for the given list of market symbols.

    :param top_markets_symbols: A list of market symbols.
    :return: A list of CoinGecko token IDs.
    """
    response = requests.get('https://api.coingecko.com/api/v3/coins/list')
    tokens = response.json()
    token_ids = []
    for t in tokens:
        if t['symbol'].lower() in [m.lower() for m in top_markets_symbols]:
            token_ids.append(t['id'])
    return token_ids


def fetch_cg_markets_data(token_ids: List[str], cg_api_key: str, days: int) -> Dict:
    """
    Fetches historical market data for a list of token IDs from the CoinGecko API and returns a dictionary of prices.

    :param token_ids: A list of CoinGecko token IDs
    :param cg_api_key: CoinGecko API key
    :param days: Number of days to fetch data for
    :return: A dictionary of prices in the format {token_id: {date_str: price}}
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    total_days = (end_date - start_date).days

    prices = {}
    with tqdm(total=total_days*len(token_ids)) as pbar:
        for token_id in token_ids:
            for i in range(total_days):
                date = start_date + datetime.timedelta(days=i)
                date_str = date.strftime('%d-%m-%Y')
                url = f"https://api.coingecko.com/api/v3/coins/{token_id}/history?date={date_str}&localization=false&x_cg_pro_api_key={cg_api_key}"
                response = requests.get(url)
                while True:
                    if response.status_code == 200:
                        break
                    else:
                        time.sleep(5)
                        response = requests.get(url)
                data = response.json()
                market_price = data['market_data']['current_price']['usd']
                if token_id not in prices:
                    prices[token_id] = {}
                prices[token_id][date_str] = market_price
                pbar.set_description(desc=f"fetching prices data for {token_id} day: {i}")
                pbar.update(1)
    return prices


def fetch_binance_markets_data(token_ids: List[str], days: int, interval: str, api_key: str) -> Dict:
    """
    Fetches historical OHLCV data from the Binance API for a list of token pairs.

    :param token_ids: A list of token pairs to fetch data for (e.g. ['ETH', 'BTC']).
    :param days: Number of days of data to fetch.
    :param interval: The interval to fetch data for (e.g. '1h', '4h', '1d', etc.).
    :param api_key: Binance API key for authentication.

    :return: A dictionary of historical OHLCV data for the specified tokens and time interval.
    """
    if 'h' in interval:
        interval_length = int(interval.replace('h', ''))
        unit = 'hour'
        
    end_timestamp = int(datetime.datetime.now().timestamp() * 1000)
    start_timestamp = int((datetime.datetime.now() - datetime.timedelta(days=days)).timestamp() * 1000)
    
    prices = {}
    with tqdm(total=len(token_ids), position=0, leave=True) as pbar:
        for token_id in token_ids:
            url = f"https://api.binance.com/api/v3/klines?symbol={token_id}USDT&interval={interval}&startTime={start_timestamp}&endTime={end_timestamp}&limit=1000"
            headers = {'X-MBX-APIKEY': api_key}
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                token_prices = {}
                for d in data:
                    timestamp = int(d[0])
                    date_time = datetime.datetime.fromtimestamp(timestamp/1000.0)
                    date_str = date_time.strftime('%Y-%m-%d')
                    if unit == 'hour':
                        time_str = date_time.strftime('%H')
                        hour = int(time_str) // interval_length * interval_length
                        time_str = f"{hour:02d}:00"
                    close_price = float(d[4])
                    token_prices[f"{date_str} {time_str}"] = close_price
                prices[token_id] = token_prices
                
            pbar.set_description(desc=f"fetching prices data for {token_id}")
            time.sleep(0.1)
            pbar.update(1)

    return prices


def prices_dict_to_df(prices: Dict, intervals_count: int, fillna: bool) -> pd.DataFrame:
    """
    Converts a dictionary containing prices data to a Pandas DataFrame.

    :param prices: a dictionary containing prices data
    :param intervals_count: the number of intervals to include in the output DataFrame
    :param fillna: a boolean flag indicating whether to fill NaN values with the mean value of the column
    :return: a Pandas DataFrame containing the prices data
    """
    dfs_list = []
    for k, v in prices.items():
        if len(v) > 0:
            dfs_list.append(pd.DataFrame({k: v}))
    df_prices = pd.concat(dfs_list, axis=1)
    df_prices.index.name = 'timestamp'
    df_prices = df_prices.iloc[-intervals_count:, :]
    if fillna:
        df_prices.fillna(df_prices.mean(), inplace=True)
    return df_prices.iloc[-intervals_count:, :]