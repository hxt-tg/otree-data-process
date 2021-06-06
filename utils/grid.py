def ask_W_H(num_players):
    while True:
        w = h = int(num_players ** 0.5)
        print(f'Input width and height, split with comma. (default "W,H = {w},{h}")')
        try:
            ans = input("W,H = ")
            if len(ans) > 0: w, h = map(int, ans.replace(' ', '').split(' '))
            assert w * h == num_players
            return w, h
        except Exception:
            pass


def neighbor_mapping(w: int, h: int):
    nei = dict()

    for i in range(h):
        for j in range(w):
            nei[i * w + j + 1] = dict(L=(i * w + (j - 1) % w) + 1,
                                      U=((i - 1) % h * w + j) + 1,
                                      R=(i * w + (j + 1) % w) + 1,
                                      D=((i + 1) % h * w + j) + 1)
    return nei
