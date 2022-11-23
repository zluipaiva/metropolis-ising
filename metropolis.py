from numba import jit
import numpy as np

def init(N):
    return np.random.choice([-1, 1], N)


def gen_neigh(L):
    N = L**2

    neigh = np.empty((N, 4), dtype=np.int16)

    for i in range(N):
        if i % L == L - 1:  # dir
            right = i - L + 1
        else:
            right = i + 1

        neigh[i][0] = right

        if i >= (L**2) - L:  # cima
            up = i - N + L
        else:
            up = i + L

        neigh[i][1] = up

        if i % L == 0:  # esq
            left = i + L - 1
        else:
            left = i - 1

        neigh[i][2] = left

        if i < L:  # baixo
            down = i + N - L
        else:
            down = i - L

        neigh[i][3] = down

    return neigh


@jit(nopython=True)
def calc_energy(spins, neigh):
    N = len(spins)
    energy = 0

    for i in range(N):
        neigh_i = neigh[i][0:2]

        for j in neigh_i:
            energy -= spins[i] * spins[j]

    mag = np.sum(spins)
    return energy, mag


# energy difference when i is flipped
@jit(nopython=True)
def en_diff(i, spins, neigh):
    sum = 0

    for j in neigh[i]:
        sum += spins[j]

    delta = 2 * spins[i] * sum

    return delta


@jit(nopython=True)
def get_expos(T):
    expos = np.zeros(5, dtype=np.float32)
    expos[0] = np.exp(8 / T)
    expos[1] = np.exp(4 / T)
    expos[2] = 1
    expos[3] = np.exp(-4 / T)
    expos[4] = np.exp(-8 / T)

    return expos


@jit(nopython=True)
def mc_step(spins, energy, mag, neigh, expos):
    N = len(spins)
    for i in range(N):
        delta_e = en_diff(i, spins, neigh)

        de = int(delta_e*0.25 + 2)
        P = expos[de]
        r = np.random.rand()

        if r <= P:
            spins[i] = -spins[i]
            energy += delta_e
            mag = mag + 2*spins[i]

    return spins, energy, mag

def calc_err(qtty, qtty_boxes, n_boxes):
    sum = 0

    for i in range(n_boxes):
        sum += (qtty - qtty_boxes[i])**2

    return np.sqrt(sum / (n_boxes * (n_boxes - 1)))
