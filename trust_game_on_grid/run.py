import os
from os.path import join
import sys
from openpyxl import Workbook
import pandas as pd
from typing import List

sys.path.append('..')
from utils.io import OTreeData
from utils import option_choice

DATA_DIR = 'data'
OUTPUT_DIR = 'output'


def init():
    if os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def select_data_file():
    is_valid_data = lambda x: not x.startswith('~$') and (x.endswith('.xlsx') or x.endswith('.csv'))
    files = []
    for name in filter(is_valid_data, os.listdir(DATA_DIR)):
        files.append(name)
    if len(files) == 0:
        raise RuntimeError('No valid data file found.')
    elif len(files) == 1:
        return join(DATA_DIR, files[0])
    else:
        return join(DATA_DIR, files[
            option_choice(files, "Choose one which you want to process:")])


def load_session(file_name):
    data = OTreeData(file_name)
    session_codes = data.session_codes()
    if len(session_codes) == 0:
        raise RuntimeError("No valid sessions.")
    elif len(session_codes) == 1:
        return session_codes[0], data.get_session()
    else:
        option_string = [
            f'{s} ({data.num_players(s)} players, {data.num_rounds(s)} rounds, '
            f'start at {data.time_started(s)}, data {"in" if not data.is_data_complete(s) else ""}complete)'
            for s in session_codes
        ]
        session_code = option_choice(option_string, "Choose one session below:")
        return session_code, data.get_session(session_codes[session_code])


def save_player_round(session_code, data: pd.DataFrame):
    """Save player-rounds matrix."""
    def _save_total(ws, d: pd.Series, col_idx):
        for i in range(len(d)):
            ws.cell(row=i+2, column=col_idx, value=d.iloc[i])

    def _save_mat(ws, d: pd.DataFrame):
        ws.cell(row=1, column=1, value='id\\round')
        for i in range(1, d.shape[0]+1):
            ws.cell(row=i+1, column=1, value=i)
        for i in range(1, d.shape[1]+1):
            ws.cell(row=1, column=i+1, value=i)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                ws.cell(row=i+2, column=j+2, value=d.iloc[i, j])
    file_name = join(OUTPUT_DIR, f'trust_game_on_grid_{session_code}.xlsx')
    wb = Workbook()
    ws0 = wb.active
    ws0.title = "All"
    ws0.cell(row=1, column=1, value='round')
    ws0.cell(row=1, column=2, value='id')
    ws0.cell(row=1, column=3, value='send_T')
    ws0.cell(row=1, column=4, value='return_T')
    ws0.cell(row=1, column=5, value='receive_send_T')
    ws0.cell(row=1, column=6, value='receive_return_T')

    for _i, (round_number, player_id) in enumerate(data.index):
        ws0.cell(row=_i+2, column=1, value=round_number)
        ws0.cell(row=_i+2, column=2, value=player_id)

    _save_total(ws0, data.player.send_T, 3)
    _save_total(ws0, data.player.return_T, 4)
    _save_total(ws0, data.player.receive_send_T, 5)
    _save_total(ws0, data.player.receive_return_T, 6)
    _save_mat(wb.create_sheet(title='send_T'), data.player.send_T.unstack().T)
    _save_mat(wb.create_sheet(title='return_T'), data.player.return_T.unstack().T)
    _save_mat(wb.create_sheet(title='receive_send_T'), data.player.receive_send_T.unstack().T)
    _save_mat(wb.create_sheet(title='receive_return_T'), data.player.receive_return_T.unstack().T)
    wb.save(file_name)


OPERATION = [
    save_player_round
]


def run():
    file_name = select_data_file()
    session_code, data = load_session(file_name)
    OPERATION[option_choice(list(map(lambda x: x.__doc__, OPERATION)), "Choose one operation:")](session_code, data)
    # save_player_round(session_code, data)


if __name__ == '__main__':
    run()
