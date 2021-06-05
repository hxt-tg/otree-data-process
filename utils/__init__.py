from typing import List


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
