from os.path import join
import sys
from openpyxl import Workbook
import pandas as pd

sys.path.append('..')
from utils import option_choice, init, select_data_file, load_session

DATA_DIR = 'data'
OUTPUT_DIR = 'output'
APP_NAME = 'trust_game_on_grid'


def save_player_round(session_code, data: pd.DataFrame):
    """Save player-rounds matrix."""

    def _save_raw_data(ws, fields):
        d = data.loc[:, list(map(lambda x: ('player', x), fields))]
        ws.append(['round', 'id'] + fields)
        for _round_number, _player_id in d.index:
            ws.append([_round_number, _player_id] + list(d.loc[(_round_number, _player_id), :].values))

    def _save_mat(ws, d: pd.DataFrame):
        ws.cell(row=1, column=1, value='id\\round')
        for i in range(1, d.shape[0] + 1):
            ws.cell(row=i + 1, column=1, value=i)
        for i in range(1, d.shape[1] + 1):
            ws.cell(row=1, column=i + 1, value=i)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                ws.cell(row=i + 2, column=j + 2, value=d.iloc[i, j])

    file_name = join(OUTPUT_DIR, f'{APP_NAME}_{session_code}.xlsx')
    wb = Workbook()
    ws0 = wb.active
    ws0.title = "All"

    _save_raw_data(ws0, ['send_T', 'return_T', 'receive_send_T', 'receive_return_T', 'payoff', 'is_node_player'])
    _save_mat(wb.create_sheet(title='send_T'), data.player.send_T.unstack().T)
    _save_mat(wb.create_sheet(title='return_T'), data.player.return_T.unstack().T)
    _save_mat(wb.create_sheet(title='receive_send_T'), data.player.receive_send_T.unstack().T)
    _save_mat(wb.create_sheet(title='receive_return_T'), data.player.receive_return_T.unstack().T)
    _save_mat(wb.create_sheet(title='payoff'), data.player.payoff.unstack().T)
    wb.save(file_name)


OPERATION = [
    save_player_round
]


def run():
    init(DATA_DIR, OUTPUT_DIR)
    file_name = select_data_file(DATA_DIR)
    session_code, data = load_session(file_name)
    OPERATION[option_choice(list(map(lambda x: x.__doc__, OPERATION)), "Choose one operation:")](session_code, data)
    # save_player_round(session_code, data)


if __name__ == '__main__':
    run()
