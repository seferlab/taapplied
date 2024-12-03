from architectures.helpers.constants import threshold
from architectures.helpers.model_handler import get_model
from architectures.helpers.constants import hyperparameters
from architectures.helpers.constants import etf_list
from architectures.helpers.constants import threshold
from architectures.helpers.constants import selected_model

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

hyperparameters = hyperparameters[selected_model]


class Wallet:
    def __init__(self, base_currency_name: str, stock_name: str, initial_money: float):
        self.base_currency_name: str = base_currency_name
        self.stock_name: str = stock_name
        self.initial_money: float = initial_money
        self.info: dict = {base_currency_name: initial_money, stock_name: 0, f"v_{base_currency_name}": initial_money, f"v_{stock_name}": 0,
                           "buy_count": 0, "hold_count": 0, "sell_count": 0}
        self.profit_percentage: float = 0
        #self.transactions: list = []

    def buy(self, stock_price: float, date: str):
        if self.info[self.base_currency_name] == 0:
            return
        self.info["buy_count"] += 1
        v_base = (self.info[self.base_currency_name] - 1)
        stock = v_base / stock_price
        # print(
        #     f"Bought {self.stock_name}: {round(stock, 2)} | USD: 0 | price: {round(stock_price, 2)} | date: {date}")
        self.info[self.stock_name] = stock
        self.info[f"v_{self.stock_name}"] = stock
        self.info[self.base_currency_name] = 0
        self.info[f"v_{self.base_currency_name}"] = v_base
        self.profit_percentage = v_base / self.initial_money - 1

    def hold(self, stock_price: float):
        self.info["hold_count"] += 1
        self.update_values(stock_price)
        return

    def sell(self, stock_price: float, date: str):
        if self.info[self.stock_name] == 0:
            return
        self.info["sell_count"] += 1
        base = self.info[self.stock_name] * stock_price - 1
        v_stock = base / stock_price
        # print(
        #     f"Sold   {self.stock_name}: 0 | USD: {round(base, 2)} | price: {round(stock_price, 2)} | date: {date}")
        self.info[self.base_currency_name] = base
        self.info[f"v_{self.base_currency_name}"] = base
        self.info[self.stock_name] = 0
        self.info[f"v_{self.stock_name}"] = v_stock
        self.profit_percentage = base / self.initial_money - 1

    #def print_values(self):
    #    # if(self.profit_percentage > 0):
    #    print(self.info)
    #    print(f"Profit percentage: {self.profit_percentage/4}")

    def update_values(self, stock_price: float):
        if self.info[self.stock_name] > 0:
            self.info[f"v_{self.base_currency_name}"] = self.info[self.stock_name] * stock_price
        elif self.info[self.base_currency_name] > 0:
            self.info[f"v_{self.stock_name}"] = self.info[self.base_currency_name] / stock_price
        else:
            print("Error")
        self.profit_percentage = self.info[f"v_{self.base_currency_name}"] / \
            self.initial_money - 1


def load_dataset():
    x_test = []
    y_test = []
    for etf in etf_list:
        x_test.append(np.load(f"ETF/TestData/x_{etf}.npy"))
        y_test.append(np.load(f"ETF/TestData/y_{etf}.npy"))
    return x_test, y_test


def make_dataset(x_test, y_test):
    datasets = []
    # keeps the images and labels for every stock one by one (datasets[0] == images & labels for etf_list[0])
    for xt, yt in zip(x_test, y_test):
        dataset = tf.data.Dataset.from_tensor_slices((xt, yt))
        dataset = dataset.batch(hyperparameters["batch_size"])
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        datasets.append(dataset)
    return datasets


def calculate_max_drawdown(returns):
    """
    Calculate the maximum drawdown from a series of returns.

    Parameters:
        returns (list or np.ndarray): Series of periodic returns.

    Returns:
        max_drawdown (float): Maximum drawdown as a percentage.
        drawdown_start (int): Index of the peak before the drawdown starts.
        drawdown_end (int): Index of the trough at the end of the drawdown.
    """
    # Convert returns to cumulative returns
    cumulative_returns = np.cumprod(1 + np.array(returns))
    
    # Track the running maximum
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Drawdown is the percentage drop from the running maximum
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Maximum drawdown
    max_drawdown = drawdown.min()
    
    # Find the indices of the max drawdown
    drawdown_end = np.argmin(drawdown)  # Index of the lowest point
    drawdown_start = np.argmax(cumulative_returns[:drawdown_end + 1])  # Peak before the drawdown
    
    return max_drawdown, drawdown_start, drawdown_end


def portfolioAnalysis():
    """
    """
    etf2dates, etf2prices = {}, {}
    for etf in etf_list:
        etf2dates[etf] = np.load(f"ETF/Date/TestDate/{etf}.npy", allow_pickle=True)
        etf2prices[etf] = np.load(f"ETF/Price/TestPrice/{etf}.npy", allow_pickle=True)

    x_test, y_test = load_dataset()
    datasets = make_dataset(x_test, y_test)

    filepath = "my_model.weights.h5"
    model = get_model()
    model.load_weights(filepath)
    etf2signal = {}
    for ind,dataset in enumerate(datasets):
        predictions = model.predict(dataset)
        etf = etf_list[ind]
        etf2signal[etf] = np.argmax(predictions, axis=1)

    use_dates = sorted(list(etf2dates.values())[0])
    daily_moneys = [100]
    returns = []
    date2trades = {}
    prev_portfolio = {}
    for ind,date in enumerate(use_dates):
        longs = []
        for etf in etf_list:
            if etf2signal[etf][ind] == 0:
                longs.append(etf)

        if ind == len(use_dates) - 1:
            break
        
        if len(longs) == 0:
            returns.append(0.0)
            daily_moneys.append(daily_moneys[-1])
            continue
        
        alloc = daily_moneys[-1] / len(longs)
        portfolio = {}
        for etf in longs:
            portfolio[etf] = alloc/etf2prices[etf][ind]

        univ = portfolio.keys() | prev_portfolio.keys()
        trades = {}
        for etf in univ:
            count1,count2 = 0,0
            if etf in portfolio:
                count1 = portfolio[etf]
            if etf in prev_portfolio:
                count2 = prev_portfolio[etf]
            diff = count1 - count2
            if abs(diff) < 0.01:
                trades[etf] = diff
        date2trades[date] = dict(trades)
        prev_portfolio = portfolio
        
        current_value = 0
        for etf in portfolio.keys():
            current_value += portfolio[etf] * etf2prices[etf][ind+1]

        returns.append(current_value / daily_moneys[-1] - 1)
        daily_moneys.append(current_value)
        
    print(daily_moneys)            
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = math.sqrt(252) * mean_return / std_return
    drawdown, drawdawn_start_index, drawdown_end_index = calculate_max_drawdown(returns)
    print(sharpe_ratio)
    print(mean_return)
    print(std_return)
    
    print(drawdown)

    # plot daily moneys
    plotpath = "predictions_portfolio.png"
    plt.plot(use_dates, daily_moneys)
    plt.savefig(plotpath, dpi=600)
    
    output = {
        "Sharpe Ratio" : sharpe_ratio,
        "Drawdown" : drawdown,
        "Mean Return" : mean_return,
        "Std Return" : std_return,
        "Dates": use_dates,
        "Portfolio": daily_moneys,
        "Returns": returns,
        "Trades" : date2trades,
    }
    return output

def singleetf_analyze(etf_selected):
    """ Analyze single ETF
    """
    listOfDates = []
    listOfPrices = []
  
    for etf in etf_list:
        listOfDates.append(np.load(f"ETF/Date/TestDate/{etf}.npy", allow_pickle=True))
        listOfPrices.append(np.load(f"ETF/Price/TestPrice/{etf}.npy", allow_pickle=True))

    x_test, y_test = load_dataset()
    datasets = make_dataset(x_test, y_test)

    filepath = "my_model.weights.h5"
    model = get_model()
    model.load_weights(filepath)
    listOfSignals = []
    for dataset in datasets:
        predictions = model.predict(dataset)
        listOfSignals.append(np.argmax(predictions, axis=1))

    daily_moneys = None
    use_dates = None
    returns = None
    for signals, etf, price, dates in zip(listOfSignals, etf_list, listOfPrices, listOfDates):
        if etf != etf_selected:
            continue
        
        wallet = Wallet("USD", etf, 100)
        daily_money = []
        for signal, price, date in zip(signals, price, dates):
            if signal == 0:
                wallet.buy(price, date)
            elif signal == 1:
                wallet.hold(price)
            elif signal == 2:
                wallet.sell(price, date)
            daily_money.append(wallet.info[f"v_{wallet.base_currency_name}"])
        #wallet.print_values()
        use_dates = list(dates)

        local_returns = []
        for ind in range(1,len(daily_money)):
            ret = daily_money[ind] / daily_money[ind-1] - 1.0
            local_returns.append(ret)
        
        daily_moneys = daily_money
        returns = local_returns

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = math.sqrt(252) * mean_return / std_return
    drawdown, drawdawn_start_index, drawdown_end_index = calculate_max_drawdown(returns)
    print(mean_return)
    print(std_return)
    print(sharpe_ratio)
    print(drawdown)
    
    # plot daily moneys
    plotpath = "predictions_{0}.png".format(etf_selected)
    plt.plot(use_dates, daily_moneys)
    plt.savefig(plotpath, dpi=600)

    output = {
        "Sharpe Ratio" : sharpe_ratio,
        "Drawdown" : drawdown,
        "Mean Return" : mean_return,
        "Std Return" : std_return,
        "Dates": use_dates,
        "Portfolio": daily_moneys,
        "Returns": returns
    }
    return output
    
#singleetf_analyze("QQQ")
portfolioAnalysis()
