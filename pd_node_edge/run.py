from os.path import join
import os
import sys
from openpyxl import Workbook
import pandas as pd
import numpy as np

sys.path.append('..')
from utils import option_choice, init, select_data_file, load_session, grid
from utils.io import pd_to_sheet, player_round_mat_to_sheet

DEBUG = False
DFT_W, DFT_H = 7, 7
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
APP_NAME = 'pd_node_edge'
I_ROUND = 0
I_ID = 1


def save_neighbor_info(session_code, data: pd.DataFrame, w, h):
    """Save a excel with all neighbors' information."""

    file_name = join(OUTPUT_DIR, f'{APP_NAME}_neighbors_{session_code}.xlsx')
    wb = Workbook()

    ws0 = wb.active
    ws0.title = "All"
    pd_to_sheet(ws0, data, 'type payoff pid_L pid_U pid_R pid_D '
                           'choice_L choice_U choice_R choice_D '
                           'type_L type_U type_R type_D '
                           'choice_nei_L choice_nei_U choice_nei_R choice_nei_D'.split(' '))
    wb.save(file_name)


OPERATION = [
    save_neighbor_info
]


def fill_edge_choice(data: pd.DataFrame):
    nodes = data.player.node_choice.notna()
    data.loc[nodes,
             [('player', 'choice_L'),
              ('player', 'choice_U'),
              ('player', 'choice_R'),
              ('player', 'choice_D')
              ]] = data.loc[nodes, ('player', 'node_choice')]
    data.loc[:, ('player', 'type')] = np.where(nodes, 'Node', 'Edge')
    return data[data.player.choice_L.notna()]


def calculate_data(data: pd.DataFrame, w, h):
    nei = grid.neighbor_mapping(w, h)
    data = data.reindex(columns=data.columns.tolist()
                                + list(map(lambda x: ('player', x),
                                           'pid_L pid_U pid_R pid_D '
                                           'type_L type_U type_R type_D '
                                           'choice_nei_L choice_nei_U choice_nei_R choice_nei_D'.split(' '))))
    for idx, row in data.iterrows():
        rnd, pid = idx
        print(f'\rProducing neighbor info (round {rnd}, player {pid}) ...    ', end='')
        data.loc[idx, ('player', 'pid_L')] = nei[pid]['L']
        data.loc[idx, ('player', 'pid_U')] = nei[pid]['U']
        data.loc[idx, ('player', 'pid_R')] = nei[pid]['R']
        data.loc[idx, ('player', 'pid_D')] = nei[pid]['D']
        data.loc[idx, ('player', 'type_L')] = data.loc[(rnd, nei[pid]['L']), ('player', 'type')]
        data.loc[idx, ('player', 'type_U')] = data.loc[(rnd, nei[pid]['U']), ('player', 'type')]
        data.loc[idx, ('player', 'type_R')] = data.loc[(rnd, nei[pid]['R']), ('player', 'type')]
        data.loc[idx, ('player', 'type_D')] = data.loc[(rnd, nei[pid]['D']), ('player', 'type')]
        data.loc[idx, ('player', 'choice_nei_L')] = data.loc[(rnd, nei[pid]['L']), ('player', 'choice_R')]
        data.loc[idx, ('player', 'choice_nei_U')] = data.loc[(rnd, nei[pid]['U']), ('player', 'choice_D')]
        data.loc[idx, ('player', 'choice_nei_R')] = data.loc[(rnd, nei[pid]['R']), ('player', 'choice_L')]
        data.loc[idx, ('player', 'choice_nei_D')] = data.loc[(rnd, nei[pid]['D']), ('player', 'choice_U')]
    print('Done')
    return data


def run():
    init(DATA_DIR, OUTPUT_DIR)
    file_name = select_data_file(DATA_DIR)
    session_code, data = load_session(file_name)
    data = data.rename(
        columns=dict(left_edge='choice_L', up_edge='choice_U', right_edge='choice_R', down_edge='choice_D'))
    num_players = len(data.iloc[data.index.get_level_values('round_number') == 1])
    w, h = grid.ask_W_H(num_players) if not DEBUG else (DFT_W, DFT_H)
    data = fill_edge_choice(data)
    data = calculate_data(data, w, h)
    if not DEBUG:
        while True:
            OPERATION[option_choice(list(map(lambda x: x.__doc__, OPERATION)), "Choose one operation:")](
                session_code, data, w, h)
    else:
        save_neighbor_info(session_code, data, w, h)


if __name__ == '__main__':
    run()
    if not DEBUG:
        os.system("pause")
