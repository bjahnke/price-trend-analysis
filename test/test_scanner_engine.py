import pytest
import pandas as pd
import src.trend_analysis.scanner_engine as scanner_engine
import env


def test_new_regime_scanner():
    res = scanner_engine.new_regime_scanner(
        [0, 1, 2, 3],
        env.NEON_DB_CONSTR,
        *(
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
    )
    print('done')
