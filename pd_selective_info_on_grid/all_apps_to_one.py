from os.path import join
import os
import sys
import traceback
import pandas as pd

sys.path.append('..')

DEBUG = False
DFT_W, DFT_H = 6, 6
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
APP_NAME = 'pd_selective_info_on_grid'
I_ROUND = 0
I_ID = 1

global_vars = dict()

REPEAT_COLS = [
    'participant.id_in_session',
    'participant.code',
    'participant.label',
    'participant._is_bot',
    'participant._index_in_pages',
    'participant._max_page_index',
    'participant._current_app_name',
    'participant._current_page_name',
    'participant.time_started',
    'participant.visited',
    'participant.mturk_worker_id',
    'participant.mturk_assignment_id',
    'participant.payoff',
    'session.code',
    'session.label',
    'session.mturk_HITId',
    'session.mturk_HITGroupId',
    'session.comment',
    'session.is_demo']


def save_per_app(file_name):
    input_file_name = join(DATA_DIR, file_name)
    output_file_name = join(DATA_DIR, f'transform_per_app_{file_name}')
    if input_file_name.endswith('.xlsx'):
        data = pd.read_excel(input_file_name)
    elif input_file_name.endswith('.csv'):
        data = pd.read_csv(input_file_name)
    else:
        raise ValueError('Only support `*.xlsx` and `*.csv`.')
    data_all_labels = list(filter(lambda x: x.startswith(APP_NAME), data.columns))
    data_cols = list(map(lambda x: tuple(x.split('.')), data_all_labels))
    assert len(data_cols) > 0, f"No valid {APP_NAME} data columns."

    max_rounds = max(map(lambda x: int(x[1]), data_cols))
    data_per_app_labels = list(map(lambda x: '.'.join(x[2:]), data_cols[:len(data_cols) // max_rounds]))
    cols = list(filter(lambda x: x in REPEAT_COLS or x.startswith(APP_NAME), data.columns))
    data = data.loc[data[f"{APP_NAME}.1.subsession.round_number"].notna()]
    data = data.loc[:, cols]
    session_codes = data['session.code'].unique()

    # Output
    df_result = pd.DataFrame([], columns=REPEAT_COLS + data_per_app_labels)

    def round_labels(n):
        return list(map(lambda x: f'{APP_NAME}.{n}.{x}', data_per_app_labels))

    for session_code in session_codes:
        session_data = data.loc[data['session.code'] == session_code]
        for rnd in range(1, max_rounds + 1):
            d = session_data[REPEAT_COLS + round_labels(rnd)]
            d.columns = list(map(lambda x: '.'.join(x.split('.')[-2:]), d.columns))
            df_result = df_result.append(d)
        # d_data = data[data_all_labels]
        # pd.options.display.max_columns = None
    df_result.to_csv(output_file_name, index=False)


def run():
    is_valid_data = lambda x: not x.startswith('~$') and not x.startswith('transform_per_app_') and \
                              (x.endswith('.xlsx') or x.endswith('.csv'))
    files = list(filter(is_valid_data, os.listdir(DATA_DIR)))
    for file in files:
        save_per_app(file)
        break

    exit(0)


if __name__ == '__main__':
    from warnings import warn

    warn("This version is only for `ALL APPs`, and will be rewrite or deprecated.", FutureWarning)
    try:
        run()
    except Exception:
        traceback.print_exception(*sys.exc_info())
        if not DEBUG:
            os.system("pause")
