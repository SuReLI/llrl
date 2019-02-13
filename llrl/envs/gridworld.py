import numpy as np
import sys
import llrl.utils.colorize as colorize
from llrl.spaces.discrete import Discrete as Discrete
from six import StringIO

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

MAPS = {
    "frozenlake": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "maze1": [
        "WFF",
        "GFF",
        "FFS"
    ],
    "maze2": [
        "WGF",
        "FFF",
        "FFS"
    ],
    "maze3": [
        "FFF",
        "FSF",
        "FGW"
    ]
}


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class GridWorld(object):
    """
    S : starting point
    F : floor
    H : hole, fall to your doom (terminal)
    W : wall
    G : goal (terminal)
    """

    def __init__(self, desc=None, map_name="maze", is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = desc.shape

        self.nS = self.nrow * self.ncol  # n states
        self.nA = 4  # n actions
        self.action_space = Discrete(self.nA)
        # self.nT = nT
        self.is_slippery = is_slippery
        self.tau = 1  # timestep duration
        isd = np.array(self.desc == b'S').astype('float64').ravel()  # Initial state distribution
        self.isd = isd / isd.sum()
        self.T = self.generate_transition_matrix()

        self.np_random = np.random.RandomState()
        self.state = categorical_sample(self.isd, self.np_random)
        self.last_action = None  # for rendering
        # self.reset()

    def reset(self):
        """
        Reset the environment.
        IMPORTANT: Does not create a new environment.
        """
        self.state = categorical_sample(self.isd, self.np_random)
        self.last_action = None  # for rendering
        return self.state

    def display(self):
        print('Displaying GridWorld')
        print('map       :')
        print(self.desc)
        print('n states  :', self.nS)
        print('n actions :', self.nA)
        # print('timeout   :', self.nT)

    def display_to_m(self, v):
        for i in range(self.nrow):
            print(v[i * self.ncol:(i + 1) * self.ncol])

    def inc(self, row, col, a):
        """
        Given a position (row, col) and an action a, return the resulting position (row, col).
        """
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return row, col

    def to_s(self, row, col):
        """
        From the state's position (row, col), retrieve the state index.
        """
        return row * self.ncol + col

    def to_m(self, s):
        """
        From the state index, retrieve the state's position (row, col).
        """
        row = int(s / self.ncol)
        col = s - row * self.ncol
        return row, col

    def distance(self, s1, s2):
        row1, col1 = self.to_m(s1)
        row2, col2 = self.to_m(s2)
        return abs(row1 - row2) + abs(col1 - col2)

    def equality_operator(self, s1, s2):
        """
        Return True if the input states have the same indexes.
        """
        return s1 == s2

    def reachable_states(self, s, a):
        rs = np.zeros(shape=self.nS, dtype=int)
        if self.is_terminal(s):  # self-loop
            rs[s] = 1
        else:
            row, col = self.to_m(s)
            if self.is_slippery:
                for b in [(a - 1) % 4, a, (a + 1) % 4]:
                    nr, nc = self.inc(row, col, b)
                    letter = self.desc[nr, nc]
                    if bytes(letter) in b'W':
                        rs[self.to_s(row, col)] = 1
                    else:
                        rs[self.to_s(nr, nc)] = 1
            else:
                nr, nc = self.inc(row, col, a)
                letter = self.desc[nr, nc]
                if bytes(letter) in b'W':
                    rs[self.to_s(row, col)] = 1
                else:
                    rs[self.to_s(nr, nc)] = 1
        return rs

    def distances_matrix(self, states):
        """
        Return the distance matrix D corresponding to the states of the input array.
        D[i,j] = distance(si, sj)
        """
        n = len(states)
        D = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dij = self.distance(states[i], states[j])
                D[i, j] = dij
                D[j, i] = dij
        return D

    def generate_transition_matrix(self):
        T = np.zeros(shape=(self.nS, self.nA, self.nS), dtype=float)
        for s in range(self.nS):
            for a in range(self.nA):
                rs = self.reachable_states(s, a)
                w_slip = 0.1
                w_norm = 1.0 - float(np.sum(rs)) * w_slip
                T[s, a, :] = np.asarray([0 if x == 0 else w_slip for x in rs], dtype=float)
                row, col = self.to_m(s)
                row_p, col_p = self.inc(row, col, a)
                s_p = self.to_s(row_p, col_p)
                T[s, a, s_p] += w_norm
        return T

    def transition_probability_distribution(self, s, a):
        assert s < self.nS, 'Error: state out of range: s.index={} nS={}'.format(s, nS)
        assert a < self.nA, 'Error: action out of range: a={} nA={}'.format(a, nA)
        return self.T[s, a]

    def transition_probability(self, s_p, s, a):
        assert s < self.nS, 'Error: state out of range: s.index={} nS={}'.format(s, nS)
        assert a < self.nA, 'Error: action out of range: a={} nA={}'.format(a, nA)
        assert s_p < self.nS, 'Error: state out of range: s.index={} nS={}'.format(s_p, nS)
        return self.T[s, a, s_p]

    def transition(self, s, a):
        """
        Transition operator, return the resulting state, reward and a boolean indicating
        whether the termination criterion is reached or not.
        """
        d = self.transition_probability_distribution(s, a)
        s_p = categorical_sample(d, self.np_random)
        r = self.instant_reward(s, a, s_p)
        done = self.is_terminal(s_p)
        return s_p, r, done

    def instant_reward(self, s, a, s_p):
        """
        Return the instant reward for transition s, a, s_p
        """
        row, col = self.to_m(s)
        letter = self.desc[row, col]
        row_p, col_p = self.to_m(s_p)
        letter_p = self.desc[row_p, col_p]
        if letter == b'G' or letter_p == b'G':
            return 1.0
        elif letter_p == b'H':
            return -1.0
        else:
            return 0.0

    def expected_reward(self, s, a):
        """
        Return the expected reward function at s, a
        """
        r = 0.0
        d = self.transition_probability_distribution(s, a)
        for i in range(len(d)):
            r_i = self.instant_reward(s, a, i)
            r += r_i * d[i]
        return r

    def is_terminal(self, s):
        """
        Return True if the input state is terminal.
        """
        row, col = self.to_m(s)
        letter = self.desc[row, col]
        done = bytes(letter) in b'GH'
        # if s.time + self.tau >= self.nT:  # Timeout
        #    done = True
        return done

    def step(self, a):
        s, r, done = self.transition(self.state, a)
        self.state = s
        self.last_action = a
        return s, r, done

    def render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.state // self.ncol, self.state % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = colorize.colorize(desc[row][col], "red", highlight=True)
        if self.last_action is not None:
            outfile.write("  ({})\n".format(["Up", "Right", "Down", "Left"][self.last_action]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile
