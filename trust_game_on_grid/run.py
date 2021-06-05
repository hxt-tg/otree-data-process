from os.path import join
import sys
from openpyxl import Workbook
import pandas as pd

sys.path.append('..')
from utils import option_choice, init, select_data_file, load_session

DATA_DIR = 'data'
OUTPUT_DIR = 'output'


def save_player_round(session_code, data: pd.DataFrame):
    """Save player-rounds matrix."""

    def _save_total(ws, d: pd.Series, col_idx):
        for i in range(len(d)):
            ws.cell(row=i + 2, column=col_idx, value=d.iloc[i])

    def _save_mat(ws, d: pd.DataFrame):
        ws.cell(row=1, column=1, value='id\\round')
        for i in range(1, d.shape[0] + 1):
            ws.cell(row=i + 1, column=1, value=i)
        for i in range(1, d.shape[1] + 1):
            ws.cell(row=1, column=i + 1, value=i)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                ws.cell(row=i + 2, column=j + 2, value=d.iloc[i, j])

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
        ws0.cell(row=_i + 2, column=1, value=round_number)
        ws0.cell(row=_i + 2, column=2, value=player_id)

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
    init(DATA_DIR, OUTPUT_DIR)
    file_name = select_data_file(DATA_DIR)
    session_code, data = load_session(file_name)
    OPERATION[option_choice(list(map(lambda x: x.__doc__, OPERATION)), "Choose one operation:")](session_code, data)
    # save_player_round(session_code, data)


if __name__ == '__main__':
    run()
