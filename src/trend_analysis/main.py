from flask import Flask

app = Flask(__name__)


@app.route('/', methods=['POST'])
def handle_request():
    """
    This is the main function that will be called by the cloud function.
    :return:
    """
    return 'Service executed with no errors'


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
    connection_string = "postgresql://bjahnke71:8mwXTCZsA6tn@ep-spring-tooth-474112-pooler.us-east-2.aws.neon.tech/historical-stock-data"
    engine = create_engine(connection_string, echo=True)
    symbols = pd.read_sql('SELECT symbol FROM stock_data', engine)
    # engine.execute('DROP TABLE IF EXISTS peak')
    # engine.execute('DROP TABLE IF EXISTS enhanced_price')
    # engine.execute('DROP TABLE IF EXISTS regime')
    # data.symbol to list of unique symbols
    symbols = symbols.symbol.unique().tolist()
    if multiprocess:
        results = init_multiprocess(regime_scanner_mp, symbols, connection_string, *trend_args)
        peak_list = []
        regime_list = []
        enhanced_price_list = []
        for peak_table, regime_table, enhanced_price_data_table, error in results:
            peak_list += [peak_table]
            regime_list += [regime_table]
            enhanced_price_list += [enhanced_price_data_table]

        pd.concat(peak_list).reset_index(drop=True).to_sql('peak', engine, if_exists='replace', index=False)
        pd.concat(regime_list).reset_index(drop=True).to_sql('regime', engine, if_exists='replace', index=False)
        pd.concat(enhanced_price_list).reset_index(drop=True).to_sql('enhanced_price', engine,
                                                                            if_exists='replace', index=False)
    else:
        results = new_regime_scanner(symbols, connection_string, *trend_args)


if __name__ == '__main__':
    main(multiprocess=True)
    app.run(host='0.0.0.0', port=8080)