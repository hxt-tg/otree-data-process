import os
from typing import List
from os.path import join
from .io import OTreeData


def option_choice(options: List[str], prompt="Choose one option below:"):
    while True:
        print('-' * 50)
        print(prompt)
        for i, opt in enumerate(options):
            print(f'  {i + 1}. {opt}')
        print('  x. Exit')
        ans = input("You choose: ").lower()
        if ans.startswith('x'): exit(-1)
        try:
            ans = int(ans)
            assert 1 <= ans <= len(options)
            return ans - 1
        except Exception:
            pass
        print('Invalid option.')


def init(data_dir, output_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def select_data_file(data_dir):
    is_valid_data = lambda x: not x.startswith('~$') and (x.endswith('.xlsx') or x.endswith('.csv'))
    files = []
    for name in filter(is_valid_data, os.listdir(data_dir)):
        files.append(name)
    if len(files) == 0:
        raise RuntimeError('No valid data file found.')
    elif len(files) == 1:
        return join(data_dir, files[0])
    else:
        return join(data_dir, files[
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
