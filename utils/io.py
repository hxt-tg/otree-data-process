import pandas as pd
from datetime import datetime


class OTreeSessionData(pd.DataFrame):
    _metadata = []

    @property
    def _constructor_expanddim(self):
        return OTreeSessionData

    @property
    def _constructor(self):
        return OTreeSessionData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def session_code(self):
        try:
            return self.iloc[0].session.code
        except Exception:
            return 'Unknown'

    def is_one_round(self):
        return "round_number" not in self.index.names

    def num_rounds(self):
        assert not self.is_one_round(), "Data contains only one round."
        return max(self.index.unique("round_number").values)

    def num_players(self):
        d = self if self.is_one_round() else self.get_round(1)
        return len(d)

    def get_round(self, round_number, drop_level=True):
        if self.is_one_round(): return self
        d = self.iloc[self.index.get_level_values('round_number') == round_number]
        if len(d) == 0:
            raise ValueError(f'Round {round_number} does not existed.')
        if drop_level: d.index = d.index.droplevel('round_number')
        return d

    def is_data_complete(self):
        d = self.participant.visited
        return d.sum() == len(d)

    def session_time_started(self):
        d = self if self.is_one_round() else self.get_round(1)
        d = d.participant.time_started.copy()
        d.dropna(inplace=True)
        if len(d) == 0: return None
        return datetime.fromisoformat(d.min())

    def session_description(self):
        if self.is_one_round():
            return f'One round in session "{self.session_code}" (with {self.num_players()} players, ' \
                   f'start at "{self.session_time_started()}", data {"" if self.is_data_complete() else "in"}complete)'
        else:
            return f'Session "{self.session_code}" (with ' \
                   f'{self.num_rounds()} rounds, {self.num_players()} players, ' \
                   f'start at "{self.session_time_started()}", data {"" if self.is_data_complete() else "in"}complete)'

    def raw_data(self):
        return pd.DataFrame.__repr__(self)


class OTreeData:
    def __init__(self, file_name: str):
        """Read oTree data file from `file_name`.
        Notice: `all_apps_wide*` data is not supported.
        :param file_name: oTree data file (ends with `.xlsx` or `.csv`)
        """
        if file_name.endswith('.xlsx'):
            data = pd.read_excel(file_name)
        elif file_name.endswith('.csv'):
            data = pd.read_csv(file_name)
        else:
            raise ValueError('Only support `*.xlsx` and `*.csv`.')

        # Reindex columns
        col_tuples = [c.split('.') for c in data.columns]
        if sum(map(lambda x: len(x) != 2, col_tuples)) > 0:
            raise RuntimeError('Invalid data file, check columns.')
        data.columns = pd.MultiIndex.from_tuples(col_tuples)
        high_level_columns = set(map(lambda x: x[0], data.columns))
        if 'player' not in high_level_columns:
            raise ValueError('You may input `all_apps_wide*`, which is not supported yet.')

        # Reindex rows
        data.index = pd.MultiIndex.from_tuples(
            list(data[[('session', 'code'),
                       ('subsession', 'round_number'),
                       ('participant', 'id_in_session')]]
                 .itertuples(index=False)), names=['session_code', 'round_number', 'player_id'])
        # Sessions
        self.session_data = dict()
        for session_code in data.index.unique("session_code").values:
            d = data.iloc[data.index.get_level_values('session_code') == session_code]
            d.index = d.index.droplevel('session_code')
            self.session_data[session_code] = OTreeSessionData(d)

    def session_codes(self):
        return list(self.session_data.keys())

    def get_session(self, session_code=None):
        codes = self.session_codes()
        if session_code is None:
            session_code = codes[0]
        else:
            codes = list(filter(lambda x: x.startswith(session_code), codes))
            if len(codes) == 0:
                raise ValueError(f'Session "{session_code}" does not existed.')
            else:
                session_code = codes[0]
        return self.session_data[session_code]

    def __repr__(self):
        return f'OtreeData(sessions={list(self.session_data.keys())!r})'

    def __getitem__(self, item):
        if isinstance(item, int):
            item = self.session_codes()[item]
        assert isinstance(item, str)
        return self.get_session(item)

    def items(self):
        return self.session_data.items()


def pd_to_sheet(work_sheet, data: OTreeSessionData, fields):
    d = data.loc[:, list(map(lambda x: ('player', x), fields))]
    work_sheet.append(['round', 'id'] + fields)
    for _round_number, _player_id in d.index:
        work_sheet.append([_round_number, _player_id] + list(d.loc[(_round_number, _player_id), :].values))


def player_round_mat_to_sheet(work_sheet, data: pd.Series):
    data = data.unstack().T
    work_sheet.cell(row=1, column=1, value='id\\round')
    for i in range(1, data.shape[0] + 1):
        work_sheet.cell(row=i + 1, column=1, value=i)
    for i in range(1, data.shape[1] + 1):
        work_sheet.cell(row=1, column=i + 1, value=i)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            work_sheet.cell(row=i + 2, column=j + 2, value=data.iloc[i, j])
