import numpy as np
from hw13_solution import RNN


def test_moving_average():
    # generate data
    data = np.random.uniform(size=(10, 1))

    # generate RNN
    from moving_average import spec
    nnet = RNN(spec)

    # apply RNN to data
    result = nnet.apply(data)

    # generate truth
    def shift(arr, n=1):
        return np.concatenate((np.zeros((n, 1)), arr[:-n, :]))
    truth = (data + shift(data)) / 2

    # compare
    if np.all(np.abs(truth - result) < 1e-7):
        print('Success.')
    else:
        print('Not quite.')


def test_random():
    # generate data
    data = np.ones((10, 2))

    # generate RNN
    from example1 import spec
    nnet = RNN(spec)

    # apply RNN
    result = nnet.apply(data)
    
    # generate truth
    truth = np.array([
        [-1.02626227],
        [-0.95136591],
        [-0.98218756],
        [-0.98431546],
        [-0.98666519],
        [-0.98744132],
        [-0.98783109],
        [-0.98801656],
        [-0.98811267],
        [-0.98816380]
    ])

    # compare
    if np.all(np.abs(truth - result) < 1e-7):
        print('Success.')
    else:
        print('Not quite.')


if __name__ == "__main__":
    test_random()
    test_moving_average()
