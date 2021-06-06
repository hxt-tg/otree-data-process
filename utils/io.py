import pandas as pd
from datetime import datetime


class OTreeData:
    def __init__(self, file_name: str):
        """Read oTree data file from `file_name`.
        Notice: `all_apps_wide*` data is not supported.
        :param file_name: oTree data file (ends with `.xlsx` or `.csv`)
        """
        if file_name.endswith('.xlsx'):
            self.data = pd.read_excel(file_name)
        elif file_name.endswith('.csv'):
            self.data = pd.read_csv(file_name)
        else:
            raise ValueError('Only support `*.xlsx` and `*.csv`.')

        # Reindex columns
        self.data.columns = pd.MultiIndex.from_tuples([c.split('.') for c in self.data.columns])
        high_level_columns = set(map(lambda x: x[0], self.data.columns))
        if 'player' not in high_level_columns:
            raise ValueError('You may input `all_apps_wide*`, which is not supported yet.')

        # Reindex rows
        self.data.index = pd.MultiIndex.from_tuples(
            list(self.data[[('session', 'code'),
                            ('subsession', 'round_number'),
                            ('participant', 'id_in_session')]]
                 .itertuples(index=False)), names=['session_code', 'round_number', 'player_id'])

    def num_rounds(self, session_code=None):
        d = self.get_session(session_code)
        return max(d.index.unique("round_number").values)

    def num_players(self, session_code=None):
        d = self.get_round(1, session_code)
        return len(d)

    def session_codes(self):
        return self.data.index.unique("session_code").values

    def is_data_complete(self, session_code=None):
        d = self.get_session(session_code).participant.visited
        return d.sum() == len(d)

    def time_started(self, session_code=None):
        d = self.get_round(1, session_code).participant.time_started
        d.dropna(inplace=True)
        if len(d) == 0: return None
        return datetime.fromisoformat(d.min())

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
        d = self.data.iloc[self.data.index.get_level_values('session_code') == session_code]
        d.index = d.index.droplevel('session_code')
        return d

    def get_round(self, round_number, session_code=None):
        d = self.get_session(session_code)
        d = d.iloc[d.index.get_level_values('round_number') == round_number]
        if len(d) == 0:
            raise ValueError(f'Round {round_number} does not existed.')
        d.index = d.index.droplevel('round_number')
        return d

    def __getattr__(self, item):
        return self.data.__getattr__(item)

    def __repr__(self):
        return repr(self.data)


def pd_to_sheet(work_sheet, data: pd.DataFrame, fields):
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
