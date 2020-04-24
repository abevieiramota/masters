from heapq import heappush, heappop

X = [
        [1., 3., 2.3, 10.],
        [1.2, 1.9, 1.7, 222],
        [3.1, 3.2, 3.4]
    ]


def f(X, n):

    explored = []
    n_lists = len(X)
    lists_size = [len(x) for x in X]

    X = [sorted(x) for x in X]

    heappush(explored, (sum(x[0] for x in X), [0]*n_lists))

    result = []
    while len(result) < n:

        _, ixs = heappop(explored)

        result.append(ixs)

        for i in range(0, n_lists):
            if len(result) == n:
                break 

            if ixs[i] == lists_size[i] - 1:
                continue

            ixs_copy = ixs[::]
            ixs_copy[i] = ixs_copy[i] + 1

            heappush(explored, (sum(x[ixs_] for ixs_, x in zip(ixs_copy, X)), ixs_copy))

    return result


resultado = f(X, 4)

print(resultado)

for ixs in resultado:
    print([X[i][ixs_] for i, ixs_ in enumerate(ixs)])

expected = [
                [1., 1.2, 3.1],
                [1., 1.2, 3.2],
                [1., 1.2, 3.4],
                [1., 1.7, 3.1]
            ]

assert len(expected) == len(resultado), 'Tamanho esperado = {}. Resultado = {}'.format(len(expected), len(resultado))

for i in range(len(expected)):
    assert expected[i] == resultado[i], '{} Esperado = {}. Resultado = {}'.format(i, expected[i], resultado[i])