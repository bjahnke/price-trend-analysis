from sqlalchemy import create_engine
import pandas as pd
import multiprocessing as mp
import typing as t
import src.floor_ceiling_regime
import regime
import numpy as np

import env


def regime_ranges(df, rg_col: str):
    """
    condense the regime column into a dataframe of regime ranges with start and end columns marked
    :param df:
    :param rg_col:
    :return:
    """
    start_col = "start"
    end_col = "end"
    loop_params = [(start_col, df[rg_col].shift(1)), (end_col, df[rg_col].shift(-1))]
    boundaries = {}
    for name, shift in loop_params:
        rg_boundary = df[rg_col].loc[
            ((df[rg_col] == -1) & (pd.isna(shift) | (shift != -1)))
            | ((df[rg_col] == 1) & ((pd.isna(shift)) | (shift != 1)))
        ]
        rg_df = pd.DataFrame(data={rg_col: rg_boundary})
        rg_df.index.name = name
        rg_df = rg_df.reset_index()
        boundaries[name] = rg_df

    boundaries[start_col][end_col] = boundaries[end_col][end_col]
    return boundaries[start_col][[start_col, end_col, rg_col]]


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [
        alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
        for i in range(wanted_parts)
    ]


def init_multiprocess(analysis_function, symbols: t.List[str], *args):
    """
    run analysis function in parallel
    :param analysis_function:
    :param symbols:
    :param args:
    :return:
    """
    # set up multiprocessing with pool
    with mp.Pool(None) as p:
        # run analysis for each symbol
        results = p.map(
            analysis_function,
            [(symbol,) + args for symbol in split_list(symbols, mp.cpu_count() - 1)]
        )
    # flatten results
    return results


def regime_sma(df, _c, st, lt):
    """
    #### regime_sma(df,_c,st,lt) ####
    bull +1: sma_st >= sma_lt , bear -1: sma_st <= sma_lt
    """
    sma_lt = df[_c].rolling(lt).mean()
    sma_st = df[_c].rolling(st).mean()
    rg_sma = np.sign(sma_st - sma_lt)
    return rg_sma


def regime_breakout(df, _h, _l, window):
    """
    #### regime_breakout(df,_h,_l,window) ####
    :param df:
    :param _h:
    :param _l:
    :param window:
    :return:
    """

    hl = np.where(df[_h] == df[_h].rolling(window).max(), 1,
                  np.where(df[_l] == df[_l].rolling(window).min(), -1, np.nan))
    roll_hl = pd.Series(index=df.index, data=hl).fillna(method='ffill')
    return roll_hl


def turtle_trader(df, _h, _l, slow, fast):
    """
    #### turtle_trader(df, _h, _l, slow, fast) ####
    _slow: Long/Short direction
    _fast: trailing stop loss
    """
    _slow = regime_breakout(df, _h, _l, window=slow)
    _fast = regime_breakout(df, _h, _l, window=fast)
    turtle = pd.Series(index=df.index,
                       data=np.where(_slow == 1, np.where(_fast == 1, 1, 0),
                                     np.where(_slow == -1, np.where(_fast == -1, -1, 0), 0)))
    return turtle


def init_trend_table(price_data, sma_kwargs, breakout_kwargs, turtle_kwargs):
    """
    #### init_trend_table(price_data, sma_kwargs, breakout_kwargs, turtle_kwargs) ####
    initialize trend table with sma, breakout, turtle
    :param price_data:
    :param sma_kwargs:
    :param breakout_kwargs:
    :param turtle_kwargs:
    :return:
    """
    _c = 'close'
    _h = 'high'
    _l = 'low'
    # 'sma' + str(_c)[:1] + str(sma_kwargs['st']) + str(sma_kwargs['lt'])
    data = price_data.copy()
    data['rg'] = regime_sma(price_data, _c, sma_kwargs['st'], sma_kwargs['lt'])
    sma_ranges = regime_ranges(data, 'rg')
    sma_ranges['type'] = 'sma'
    # + str(_h)[:1] + str(_l)[:1] + str(breakout_kwargs['slow'])
    data['rg'] = regime_breakout(price_data, _h, _l, breakout_kwargs['window'])
    bo_ranges = regime_ranges(data, 'rg')
    bo_ranges['type'] = 'bo'
    #  + str(_h)[:1] + str(turtle_kwargs['fast']) + str(_l)[:1] + str(breakout_kwargs['slow'])
    data['rg'] = turtle_trader(price_data, _h, _l, breakout_kwargs['slow'], turtle_kwargs['fast'])
    tt_ranges = regime_ranges(data, 'rg')
    tt_ranges['type'] = 'tt'
    # create dataframe with sma, tt, bo as columns
    trend_table = pd.concat([sma_ranges, bo_ranges, tt_ranges])
    trend_table['symbol'] = price_data['symbol'].iloc[0]
    trend_table['is_relative'] = price_data['is_relative'].iloc[0]
    return trend_table


def format_tables(tables: src.floor_ceiling_regime.FcStrategyTables, symbol, is_relative) -> src.floor_ceiling_regime.FcStrategyTables:
    """
    helper for formating data tables for database
    :param tables:
    :param symbol:
    :param is_relative:
    :return:
    """
    tables.peak_table['symbol'] = symbol
    tables.peak_table['is_relative'] = is_relative
    tables.enhanced_price_data['symbol'] = symbol
    tables.enhanced_price_data['is_relative'] = is_relative
    tables.regime_table['symbol'] = symbol
    tables.regime_table['is_relative'] = is_relative
    tables.regime_table['type'] = 'fc'
    return tables


def calculate_trend_data(
        symbol: str, price_data: pd.DataFrame, is_relative: bool, sma_kwargs, breakout_kwargs, turtle_kwargs
) -> t.Tuple[
        src.floor_ceiling_regime.FcStrategyTables,
        pd.DataFrame,
        t.Union[t.Tuple[str, t.Type[Exception]], None]
    ]:
    """
    Helper function to reuse the trend calculation run process for relative and absolute data
    :param symbol:
    :param price_data:
    :param is_relative:
    :param sma_kwargs:
    :param breakout_kwargs:
    :param turtle_kwargs:
    :return:
    """
    price_data = price_data.loc[price_data.is_relative == is_relative].reset_index(drop=True)
    error = None
    regimes_table = pd.DataFrame()
    try:
        data_tables = src.floor_ceiling_regime.fc_scale_strategy_live(price_data=price_data)
    except (regime.NotEnoughDataError, src.floor_ceiling_regime.NoEntriesError, KeyError) as e:
        data_tables = src.floor_ceiling_regime.FcStrategyTables(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        error = (symbol, type(e))
    else:
        data_tables = format_tables(data_tables, symbol, is_relative=is_relative)
        regimes_table = init_trend_table(data_tables.enhanced_price_data, sma_kwargs, breakout_kwargs, turtle_kwargs)

    return data_tables, regimes_table, error


def new_regime_scanner(symbols, conn_str, sma_kwargs, breakout_kwargs, turtle_kwargs):
    """
    load price data from db, scan for regimes, save results to db
    :return:
    """
    # INPUTS START


    # INPUTS END
    errors = []
    engine = create_engine(conn_str)
    peak_tables = []
    regime_tables = []
    enhanced_price_data_tables = []

    for i, symbol in enumerate(symbols):
        data = pd.read_sql(f''
                           f'SELECT * ' 
                           f'FROM stock_data '
                           f'WHERE symbol = \'{symbol}\' '
                           f'order by stock_data.bar_number asc', engine)
        if data.empty:
            print(f'No data for {symbol}')
            continue
        # print symbol and timestamp to track progress
        print(f'{i}.) {symbol} {pd.Timestamp.now()}')
        price_data = data.loc[data.symbol == symbol]

        rel_tables, rel_regimes_table, rel_error = calculate_trend_data(symbol, price_data, is_relative=True, sma_kwargs=sma_kwargs, breakout_kwargs=breakout_kwargs, turtle_kwargs=turtle_kwargs)
        abs_tables, abs_regimes_table, abs_error = calculate_trend_data(symbol, price_data, is_relative=False, sma_kwargs=sma_kwargs, breakout_kwargs=breakout_kwargs, turtle_kwargs=turtle_kwargs)

        if rel_error:
            errors.append(rel_error)
        if abs_error:
            errors.append(abs_error)

        peak_tables.extend([rel_tables.peak_table, abs_tables.peak_table])
        regime_tables.extend([
            rel_tables.regime_table,
            abs_tables.regime_table,
            rel_regimes_table,
            abs_regimes_table
        ])
        enhanced_price_data_tables.extend([rel_tables.enhanced_price_data, abs_tables.enhanced_price_data])

    return pd.concat(peak_tables).reset_index(drop=True), pd.concat(regime_tables).reset_index(drop=True), pd.concat(enhanced_price_data_tables).reset_index(drop=True), errors


def main(multiprocess: bool = False):
    """
    load price data from db, scan for regimes, save results to db
    :param multiprocess:
    :return:
    """
    trend_args = (
        {  # rg sma args
            'st': 50,
            'lt': 200
        },
        {  # rg breakout args
            'slow': 200,
            'window': 100
        },
        {  # rg turtle args
            'fast': 50
        },
    )
    # get symbols from db
    engine = create_engine(env.NEON_DB_CONSTR, echo=True)
    symbols = pd.read_sql('SELECT symbol FROM stock_data', engine)
    symbols = symbols.symbol.unique().tolist()
    if multiprocess:
        results = init_multiprocess(regime_scanner_mp, symbols, env.NEON_DB_CONSTR, *trend_args)
        peak_list = []
        regime_list = []
        enhanced_price_list = []
        for peak_table, regime_table, enhanced_price_data_table, error in results:
            peak_list += [peak_table]
            regime_list += [regime_table]
            enhanced_price_list += [enhanced_price_data_table]

        peak_table = pd.concat(peak_list).reset_index(drop=True)
        regime_table = pd.concat(regime_list).reset_index(drop=True)
        enhanced_price_data_table = pd.concat(enhanced_price_list).reset_index(drop=True)
    else:
        peak_table, regime_table, enhanced_price_data_table, error = new_regime_scanner(symbols, env.NEON_DB_CONSTR, *trend_args)

    peak_table.reset_index(drop=True).to_sql('peak', engine, if_exists='replace', index=False)
    regime_table.reset_index(drop=True).to_sql('regime', engine, if_exists='replace', index=False)
    enhanced_price_data_table.reset_index(drop=True).to_sql('enhanced_price', engine, if_exists='replace', index=False)


def regime_scanner_mp(args):
    """wrapper for multiprocess usage"""
    return new_regime_scanner(*args)


if __name__ == '__main__':
    main(multiprocess=False)
