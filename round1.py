import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, OrderedDict
import math
import numpy as np

class HistoricalVWAP:
    def __init__(self, bv=0, sv=0, bpv=0, spv=0):
        self.buy_volume = bv
        self.sell_volume = sv
        self.buy_price_volume = bpv
        self.sell_price_volume = spv

class RecordedData: 
    def __init__(self):
        self.amethyst_hvwap = HistoricalVWAP()
        self.starfruit_cache = []
        self.LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}
        self.INF = int(1e9)
        self.STARFRUIT_CACHE_SIZE = 20
        self.AME_RANGE = 2
        self.POSITION = {'AMETHYSTS' : 0, 'STARFRUIT' : 0}


class Trader:
    LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}
    INF = int(1e9)
    STARFRUIT_CACHE_SIZE = 20
    AME_RANGE = 2
    POSITION = {}

    def estimate_starfruit_price(self, cache):
        x = np.array([i for i in range(self.STARFRUIT_CACHE_SIZE)])
        y = np.array(cache)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return int(self.STARFRUIT_CACHE_SIZE * m + c)   # using int conversion instead of round yields +100 | +142


    # gets the total traded volume of each time stamp and best price
    # best price in buy_orders is the max; best price in sell_orders is the min
    # buy_order indicates orders are buy or sell orders
    def get_volume_and_best_price(self, orders, buy_order = True):
        volume = 0
        best = 0 if buy_order else self.INF

        for price, vol in orders.items():
            if buy_order:
                volume += vol
                best = max(best, price)
            else:
                volume -= vol
                best = min(best, price)

        return volume, best
    

    # given estimated bid and ask prices, market take if there are good offers, otherwise market make 
    # by pennying or placing our bid/ask, whichever is more profitable
    def calculate_orders(self, asset, order_depth, our_bid, our_ask):
        orders: list[Order] = []
        
        sell_orders = OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        buy_vol, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)

        position = self.POSITION[asset]
        limit = self.LIMIT[asset]

        # penny the current highest bid / lowest ask 
        penny_buy = best_buy_price
        penny_sell = best_sell_price

        bid_price = min(penny_buy, our_bid)
        ask_price = max(penny_sell, our_ask)

        # MARKET TAKE ASKS (buy items)
        for ask, vol in sell_orders.items():
            if position < limit and (ask <= our_bid or (position < 0 and ask == our_bid+1)): 
                num_orders = min(-vol, limit - position)
                position += num_orders
                orders.append(Order(asset, ask, num_orders))

        # MARKET MAKE BY PENNYING
        if position < limit:
            num_orders = limit - position
            orders.append(Order(asset, bid_price, num_orders))
            position += num_orders

        position = self.POSITION[asset] # RESET POSITION

        # MARKET TAKE BIDS (sell items)
        for bid, vol in buy_orders.items():
            if position > -limit and (bid >= our_ask or (position > 0 and bid+1 == our_ask)):
                num_orders = max(-vol, -limit-position)
                position += num_orders
                orders.append(Order(asset, bid, num_orders))

        # MARKET MAKE BY PENNYING
        if position > -limit:
            num_orders = -limit - position
            orders.append(Order(asset, ask_price, num_orders))
            position += num_orders 

        return orders

                      
    def calculate_vwap(self, orders):
        total_volume = sum(amount for _, amount in orders.items())
        total_price_volume = sum(price * amount for price, amount in orders.items())
        if total_volume == 0:
            return 0
        return total_price_volume / total_volume
    

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        if state.traderData == '': # first run, set up data
            data = RecordedData()
        else:
            data = jsonpickle.decode(state.traderData)

        self.LIMIT = data.LIMIT
        self.INF = data.INF
        self.STARFRUIT_CACHE_SIZE = data.STARFRUIT_CACHE_SIZE
        self.AME_RANGE = data.AME_RANGE
        self.POSITION = data.POSITION

        # update position for each asset
        for asset in state.order_depths:
            self.POSITION[asset] = state.position[asset] if asset in state.position else 0

        # add orders for each asset
        for asset in state.order_depths:
            order_depth: OrderDepth = state.order_depths[asset]
            orders: list[Order] = []
            
            # split by asset
            if asset == "AMETHYSTS":
                orders += self.calculate_orders(asset, order_depth, 10000-self.AME_RANGE, 10000+self.AME_RANGE)
            
            elif asset == "STARFRUIT":
                # keep the length of starfruit cache as STARFRUIT_CACHE_SIZE
                if len(data.starfruit_cache) == self.STARFRUIT_CACHE_SIZE:
                    data.starfruit_cache.pop(0)

                _, best_sell_price = self.get_volume_and_best_price(order_depth.sell_orders, buy_order=False)
                _, best_buy_price = self.get_volume_and_best_price(order_depth.buy_orders)

                data.starfruit_cache.append((best_sell_price+best_buy_price)/2)

                # if cache size is maxed, calculate next price and place orders
                lower_bound = -self.INF
                upper_bound = self.INF

                if len(data.starfruit_cache) == self.STARFRUIT_CACHE_SIZE:
                    lower_bound = self.estimate_starfruit_price(data.starfruit_cache)-2
                    upper_bound = self.estimate_starfruit_price(data.starfruit_cache)+2

                orders += self.calculate_orders(asset, order_depth, lower_bound, upper_bound)

            # update orders for current asset
            result[asset] = orders

        traderData = jsonpickle.encode(data)

        return result, conversions, traderData