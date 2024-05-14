from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Boolean, desc
from sqlalchemy.orm import relationship, backref, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Type, TypeVar, Generic, Optional
import pandas as pd
from sqlalchemy.sql import func
import pickle
import src.position_calculation as pc

Base = declarative_base()
T = TypeVar('T', bound=Base)

latest_regimes = f"""
    SELECT 
        r1.rg, 
        MAX(r1.start) AS max_start_1,
        r1.stock_id
    FROM 
        regime r1
    INNER JOIN (
        SELECT 
            r.stock_id, 
            MAX(r.start) AS max_start
        FROM 
            regime r
        WHERE 
            r.type = 'fc'
        GROUP BY 
            r.stock_id
    ) AS latest_regime ON r1.stock_id = latest_regime.stock_id AND r1.start = latest_regime.max_start
    GROUP BY 
        r1.rg, r1.stock_id
"""
latest_peaks_by_regime = f"""
  SELECT
    MAX(p.start) AS latest_peaks,
    p.stock_id
  FROM 
    peak p
  INNER JOIN ({latest_regimes}) AS l_regime 
  ON 
    p.stock_id = l_regime.stock_id AND 
    p."type" = l_regime.rg
  WHERE
    p.lvl = 2
  GROUP BY 
    p.stock_id
"""
max_stock_data = f"""
  SELECT 
    stock_id, 
    MAX(bar_number) AS max_bar_number
  FROM 
    stock_data
  GROUP BY 
    stock_id
"""
latest_signals = f"""
SELECT
  p_outer.*,
  max_stock_data.max_bar_number - p_outer.end AS signal_age,
  stock.symbol,
  stock.is_relative,
  stock.interval
FROM 
  peak AS p_outer
INNER JOIN ({latest_peaks_by_regime}) AS latest_peaks_by_regime 
ON 
    latest_peaks_by_regime.latest_peaks = p_outer.start AND 
    latest_peaks_by_regime.stock_id = p_outer.stock_id
INNER JOIN ({max_stock_data}) AS max_stock_data 
ON 
    p_outer.stock_id = max_stock_data.stock_id
INNER JOIN stock ON stock.id = p_outer.stock_id
WHERE 
  p_outer.lvl = 2;

"""

class Stock(Base):
    __tablename__ = 'stock'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    is_relative = Column(Boolean) # Assuming String, change if different
    interval = Column(String)  # Assuming String, change if different
    data_source = Column(String)
    market_index = Column(String)
    sec_type = Column(String)


class StockData(Base):
    __tablename__ = 'stock_data'
    bar_number = Column(Integer, primary_key=True)
    close = Column(Float)
    stock_id = Column(Integer, ForeignKey('stock.id'))
    stock = relationship("Stock", backref=backref("stock_data", uselist=False))


class TimestampData(Base):
    __tablename__ = 'timestamp_data'
    bar_number = Column(Integer, primary_key=True)
    interval = Column(String)
    timestamp = Column(DateTime)
    data_source = Column(String)


class Regime(Base):
    __tablename__ = 'regime'
    start = Column(Integer, ForeignKey('timestamp_data.bar_number'), primary_key=True)
    end = Column(Integer, ForeignKey('timestamp_data.bar_number'), primary_key=True)
    rg = Column(String)
    type = Column(String)
    stock_id = Column(Integer, ForeignKey('stock.id'), primary_key=True)
    stock = relationship("Stock", backref="regime")


class FloorCeiling(Base):
    __tablename__ = 'floor_ceiling'
    test = Column(String)
    fc_val = Column(Float)
    fc_date = Column(DateTime)
    rg_ch_date = Column(DateTime, primary_key=True)
    rg_ch_val = Column(Float)
    type = Column(String)
    stock_id = Column(Integer, ForeignKey('stock.id'))


class Peak(Base):
    __tablename__ = 'peak'
    start = Column(Integer, ForeignKey('timestamp_data.bar_number'), primary_key=True)
    end = Column(Integer, ForeignKey('timestamp_data.bar_number'), primary_key=True)
    type = Column(Integer)
    lvl = Column(Integer)
    st_px = Column(Float)
    en_px = Column(Float)
    stock_id = Column(Integer, ForeignKey('stock.id'), primary_key=True)


class Position(Base):
    __tablename__ = 'position'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    entry_price = Column(Float)
    entry_date = Column(DateTime)
    stop_loss = Column(Float)
    trailing_stop = Column(Float)
    quantity = Column(Integer)


class BaseRepository(Generic[T]):
    def __init__(self, s: Session, model: Type[T], id_attr: str = 'id'):
        self.session = s
        self.model = model
        self.id_attr = id_attr

    def get_by_id(self, id_value: int) -> Optional[T]:
        return self.session.query(self.model).filter(getattr(self.model, self.id_attr) == id_value).one_or_none()


class PositionRepository(BaseRepository[Position]):
    def __init__(self, session: Session):
        super().__init__(session, StockData, 'id')

    def get_latest_by_symbol(self, symbol: str) -> Optional[Position]:
        pass

    def get_by_symbol(self, symbol: str) -> Optional[Position]:
        with self.session as s:
            result = s.query(Position).filter(Position.symbol == symbol).order_by(desc(Position.id)).first()

        if not result:
            result = None

        return result


class PositionManager:
    def __init__(self, session, symbol, cost, stop, date, trail_amount, is_relative):
        self.session = session
        self.symbol = symbol
        self.cost = cost
        self.stop = stop

        self.date = date
        self.trail_amount = trail_amount
        self.is_relative = is_relative

        self._trail = None
        self._entry_bar_number = None
        self._direction = None
        self._stock_id = None

    def calc_quantity(self, risk, multiple=1):
        if self.cost == 0 or self.stop == 0 or self.cost == self.stop:
            return 0
        return risk / (self.stop * multiple - self.cost * multiple)

    def calc_target(self, fraction):
        assert 0 < fraction < 1, "fraction must be between 0 and 1"
        return (self.cost + fraction * self.stop - self.stop) / fraction

    @property
    def stock_id(self):
        if self._stock_id:
            return self._stock_id

        self._stock_id = int(pd.read_sql(
            f"select stock.id "
            f"from stock where "
            f"stock.symbol = '{self.symbol}' and "
            f"stock.is_relative = True and "
            f"stock.interval = '1d' ",
            self.session.bind
        ).id.iloc[-1])
        return self._stock_id

    @property
    def direction(self):
        if self._direction:
            return self._direction

        self._direction = 1 if self.cost > self.stop else -1
        return self._direction

    @property
    def entry_bar_number(self):
        """
        Get the bar_number from TimestampData where timestamp matches self.entry_date.

        Returns:
        int: The bar_number corresponding to the given timestamp.
        """
        if self._entry_bar_number:
            return self._entry_bar_number

        timestamp_data = self.session.query(TimestampData).filter(TimestampData.timestamp == self.date).first()
        if timestamp_data:
            self._entry_bar_number = timestamp_data.bar_number
        else:
            print(f"Timestamp data not found for timestamp: {self.date}")
        return self._entry_bar_number

    @property
    def trail(self):
        """
        Calculate the trailing stop based entry to current price.
        :return:
        """
        # with Session(self.engine) as s:
        #     prices = s.query(StockData).filter(StockData.bar_number >= self.entry_bar_number).close

        prices = pd.read_sql(f"SELECT * "
                             f"FROM stock_data "
                             f"WHERE stock_data.bar_number >= {self.entry_bar_number} "
                             f"and stock_data.stock_id = {self.stock_id}",
                             self.session.bind
                             ).close
        self._trail = (max(prices * self.direction) - self.trail_amount) * self.direction

        return self._trail

    @property
    def target(self):
        return float(pc.TwoLegTradeEquation.Solve.price(self.stop, self.cost, 2/3))

    def plot(self, price, multiple):
        _price = price.copy()
        _price['cost'] = self.cost * multiple
        _price['stop'] = self.stop * multiple
        _price['target'] = self.target * multiple
        _price['trail'] = self.trail * multiple
        _price['m_close'] = price['close'] * multiple
        _price[['m_close', 'target', 'trail', 'entry', 'stop']].iloc[-100:].plot(
            color=['black', 'green', 'orange', 'blue', 'red'],
            figsize=(20, 10)
        )

    def as_dict(self, fraction=2/3, risk=500, multiple=1):
        return {
            'entry_date': self.date,
            'symbol': self.symbol,
            'is_relative': self.is_relative,
            'cost': self.cost,
            'stop': self.stop,
            'trail': self.trail,
            'trail_amount': self.trail_amount,
            'target': self.calc_target(fraction),
            'quantity': self.calc_quantity(risk, multiple),
            'direction': self.direction,
            'risk': risk,
            'multiple': multiple,
            'fraction': fraction,
        }


class EntryManager:
    def __init__(self, session, entry_signal, stop_loss):
        self.session = session
        self.entry_signal = entry_signal
        self._symbol = entry_signal.symbol
        self._stop = stop_loss['st_px']
        self._trail_offset = abs(entry_signal['en_px'] - entry_signal['st_px'])
        self._trail = entry_signal['st_px']
        self._current_bar = self.current_bar()
        self._cost = self._current_bar.close
        self._date = self.session.query(TimestampData).filter(
            TimestampData.bar_number == self._current_bar.bar_number).first().timestamp
        self._stock_id = entry_signal.id
        self._is_relative = entry_signal.is_relative

    def current_bar(self):
        current_bar_number_sql = f"""
        select 
            max(stock_data.bar_number) as current_bar
        from
            stock_data
        where
            stock_data.stock_id = {self.entry_signal.stock_id}
        """
        current_bar = f"""
        select 
            stock_data.*
        from
            stock_data
        where
            stock_data.stock_id = {self.entry_signal.stock_id} and
            stock_data.bar_number = ({current_bar_number_sql})
        """
        return pd.read_sql(current_bar, self.session.bind).iloc[-1]

    def calc_quantity(self, risk, multiple=1):
        if self._cost == 0 or self._stop == 0 or self._cost == self._stop:
            return 0
        return risk / (self._cost * multiple - self._stop * multiple)

    def calc_target(self, fraction):
        assert 0 < fraction < 1, "fraction must be between 0 and 1"
        return (self._cost + fraction * self._stop - self._stop) / fraction

    def as_dict(self, risk, fraction=2/3, multiple=1):
        return {
            'entry_date': self._date,
            'symbol': self._symbol,
            'cost': self._cost,
            'stop': self._stop,
            'trail': self._trail,
            'trail_amount': self._trail_offset,
            'target': self.calc_target(fraction),
            'quantity': self.calc_quantity(risk),
            'direction': self.entry_signal['type'],
            'risk': risk,
            'multiple': multiple,
            'fraction': fraction,
            'is_relative': self._is_relative,
            'stock_id': self._stock_id,
        }


class StockRepository(BaseRepository[Stock]):
    def __init__(self, session: Session):
        super().__init__(session, Stock)
        self._signal_table = None

    @property
    def signal_table(self):
        if self._signal_table:
            return self._signal_table.copy()

        self._signal_table = pd.read_sql(latest_signals, self.session.bind).reset_index().rename(columns={'index': 'id'})
        return self._signal_table.copy()

    def get_entry_table(self, risk):
        # get all stock ids
        signal_table = self.signal_table
        all_entry_data = []
        entry_table_list = []
        for i, data in signal_table.iterrows():
            stop_losses = self.get_valid_stop_losses(data)

            if stop_losses.empty:
                stop_loss = data.copy()
            else:
                stop_loss = stop_losses.iloc[-1]
            all_entry_data.append(stop_losses)
            entry = EntryManager(self.session, data, stop_loss)

            entry_dict = entry.as_dict(risk)
            entry_dict['stock_id'] = data.stock_id
            entry_table_list.append(entry_dict)

        entry_table = pd.DataFrame.from_records(entry_table_list)


        with open('entry_data.pkl', 'wb') as f:
            pickle.dump(all_entry_data, f)

        entry_table = entry_table.merge(signal_table[["stock_id", "signal_age"]], on="stock_id", how="left")
        entry_table['r_pct'] = ((entry_table.cost - entry_table.stop) * entry_table.direction) / entry_table.cost

        return entry_table

    def get_valid_stop_losses(self, signal):
        get_stop_swing = f"""
        select
            p.*
        from
            peak p
        where
            p.stock_id = {signal.stock_id} and
            p.type = {signal.type} and
            p.start < {signal.start} and
            p.end < {signal.end} and
            p.lvl = {signal.lvl} and
            (p.st_px * {signal.type}) < ({signal.st_px} * {signal.type})
        """
        return pd.read_sql(get_stop_swing, self.session.bind)


class StockDataRepository(BaseRepository[StockData]):
    def __init__(self, session: Session):
        super().__init__(session, StockData, 'bar_number')


class TimestampDataRepository(BaseRepository[TimestampData]):
    def __init__(self, session: Session):
        super().__init__(session, TimestampData, 'bar_number')


class RegimeRepository(BaseRepository[Regime]):
    def __init__(self, session: Session):
        super().__init__(session, Regime, 'stock_id')


class FloorCeilingRepository(BaseRepository[FloorCeiling]):
    def __init__(self, session: Session):
        super().__init__(session, FloorCeiling, 'stock_id')


class PeakRepository(BaseRepository[Peak]):
    def __init__(self, session: Session):
        super().__init__(session, Peak, 'stock_id')


if __name__ == '__main__':
    # Replace 'your_database_url' with the actual database URL
    _engine = create_engine('your_database_url')
    Base.metadata.create_all(_engine)
