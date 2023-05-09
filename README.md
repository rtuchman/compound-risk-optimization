Compound Risk Optimization
This is a research project that aims to simulate the Compound protocol parameters to minimize the risk of liquidation and maximize borrow usage. The project consists of three primary sections: understanding Compound mechanics, fetching data for methodology, and building a simulation framework.

Requirements
To run this project, you'll need to have Python 3.7 or higher installed, as well as the packages listed in the requirements.txt file.

Installation
Clone this repository to your local machine.
Install the required packages by running pip install -r requirements.txt.
Usage
Get the top 1,000 borrow positions on Compound by running the query_borrow_positions() method of the CompoundSubgraphQuery class in utils.py.
Calculate the volatility of the top 10 supplied assets on Compound using either the markets query or other sources like Coingecko or Binance by running the query_top_compound_markets() method of the CompoundSubgraphQuery class in utils.py.
Generate random price trajectories using the values obtained in step 2 by initializing a VarSimulator object in var_simulator.py with the prices data.
Measure the bad debt of Compound for each price trajectory. We define bad debt as: userBadDebt = (userDebt − userCollateral) and badDebt = (userBadDebt > user ∑ 0)
Calculate the Value at Risk (VaR) of the protocol by generating N measurements of step 2, and then running the calculate_var() method of the VarSimulator class.
Visualize the results using the plot_price_trajectories() and plot_var() methods of the VarSimulator class.
Data Sources
Compound data on theGraph: https://thegraph.com/hosted-service/subgraph/graphprotocol/compound-v2
Coingecko: https://www.coingecko.com/
Binance: https://www.binance.com/
Glossary
Debt: Amount of assets borrowed from an asset pool.
Under-collateralized: An account is under-collateralized if the value of an account’s debt exceeds the value of the collateral.
Collateral factor: Maximum debt-to-collateral ratio of an asset a user may borrow. When the debt-to-collateral ratio exceeds the collateral factor, the collateral is available to be liquidated.
userBadDebt = (userDebt − userCollateral)
badDebt = (userBadDebt > user ∑ 0)
Credits
This project was created by rtuchman as a research project.
