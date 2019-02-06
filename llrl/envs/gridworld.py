import numpy as np
import sys
import llrl.utils.distribution as distribution
import llrl.utils.colorize as colorize
from llrl.spaces.discrete import Discrete as Discrete
from six import StringIO, b

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
    "maze": [
        "GFFF",
        "FFFF",
        "FFFF",
        "FFFS"
    ]
}


class State:
    """
    State class
    """

    def __init__(self, index):
        self.index = index


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class Gridworld():
    """
    S : starting point
    F : floor
    H : hole, fall to your doom (terminal)
    W : wall
    G : goal (terminal)
    """

    def __init__(self, desc=None, map_name="maze", nT=100, is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = desc.shape

        self.nS = self.nrow * self.ncol  # n states
        self.nA = 4  # n actions
        self.action_space = Discrete(self.nA)
        self.nT = nT  # timeout
        self.is_slippery = is_slippery
        self.tau = 1  # timestep duration
        isd = np.array(self.desc == b'S').astype('float64').ravel()  # Initial state distribution
        self.isd = isd / isd.sum()

        self.reachable_states(State(0), 0)  # TRM

        # Seed
        self.np_random = np.random.RandomState()
        self.reset()

    def reset(self):
        """
        Reset the environment.
        IMPORTANT: Does not create a new environment.
        """
        self.state = State(categorical_sample(self.isd, self.np_random))
        self.lastaction = None  # for rendering
        return self.state

    def display(self):
        print('Displaying Gridworld')
        print('map       :')
        print(self.desc)
        print('n states  :', self.nS)
        print('n actions :', self.nA)
        print('timeout   :', self.nT)

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
        return (row, col)

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
        row1, col1 = self.to_m(s1.index)
        row2, col2 = self.to_m(s2.index)
        return abs(row1 - row2) + abs(col1 - col2)

    def equality_operator(self, s1, s2):
        """
        Return True if the input states have the same indexes.
        """
        return s1.index == s2.index

    def reachable_states(self, s, a):
        row, col = self.to_m(s.index)
        rs = np.zeros(shape=self.nS, dtype=int)
        if self.is_slippery:
            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                nr, nc = self.inc(row, col, b)
                letter = self.desc[nr, nc]
                if not(bytes(letter) in b'W'):
                    rs[self.to_s(nr, nc)] = 1
        else:
            nr, nc = self.inc(row, col, b)
            letter = self.desc[nr, nc]
            if not(bytes(letter) in b'W'):
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
        # TODO here

    def generate_transition_matrix(self):
        T = np.zeros(shape=(self.nS, self.nA, self.nT, self.nS), dtype=float)
        for s in range(self.nS):
            for a in range(self.nA):
                # Generate distribution for t=0
                rs = self.reachable_states(s, a)
                nrs = np.sum(rs)
                w = distribution.random_tabular(size=nrs)
                wcopy = list(w.copy())
                T[s, a, 0, :] = np.asarray([0 if x == 0 else wcopy.pop() for x in rs], dtype=float)
                row, col = self.to_m(s)
                row_p, col_p = self.inc(row, col, a)
                s_p = self.to_s(row_p, col_p)
                T[s, a, 0, s_p] += 1.0  # Increase weight on normally reached state
                T[s, a, 0, :] /= sum(T[s, a, 0, :])
                states = []
                for k in range(len(rs)):
                    if rs[k] == 1:
                        states.append(State(k, 0))
                D = self.distances_matrix(states)
                # Build subsequent distributions st LC constraint is respected
                for t in range(1, self.nT):  # t
                    w = distribution.random_constrained(w, D, self.L_p * self.tau)
                    wcopy = list(w.copy())
                    T[s, a, t, :] = np.asarray([0 if x == 0 else wcopy.pop() for x in rs], dtype=float)
        return T

    def transition_probability_distribution(self, s, t, a):
        assert s.index < self.nS, 'Error: index bigger than nS: s.index={} nS={}'.format(s.index, nS)
        assert t < self.nT, 'Error: time bigger than nT: t={} nT={}'.format(t, self.nT)
        assert a < self.nA, 'Error: action bigger than nA: a={} nA={}'.format(a, nA)
        return self.T[s.index, a, t]

    def transition_probability(self, s_p, s, t, a):
        assert s_p.index < self.nS, 'Error: position bigger than nS: s_p.index={} nS={}'.format(s_p.index, nS)
        assert s.index < self.nS, 'Error: position bigger than nS: s.index={} nS={}'.format(s.index, nS)
        assert t < self.nT, 'Error: time bigger than nT: t={} nT={}'.format(t, self.nT)
        assert a < self.nA, 'Error: action bigger than nA: a={} nA={}'.format(a, nA)
        return self.T[s.index, a, t, s_p.index]

    def get_time(self):
        return self.state.time

    def dynamic_reachable_states(self, s, a):
        """
        Return a numpy array of the reachable states.
        Dynamic means that time increment is performed.
        """
        rs = self.reachable_states(s, a)
        srs = []
        for i in range(len(rs)):
            if rs[i] == 1:
                srs.append(State(i, s.time + self.tau))
        assert (len(srs) == sum(rs))
        return np.array(srs)

    def static_reachable_states(self, s, a):
        """
        Return a numpy array of the reachable states.
        Static means that no time increment is performed.
        """
        rs = self.reachable_states(s, a)
        srs = []
        for i in range(len(rs)):
            if rs[i] == 1:
                srs.append(State(i, s.time))
        assert (len(srs) == sum(rs))
        return np.array(srs)

    def transition(self, s, a, is_model_dynamic=True):
        """
        Transition operator, return the resulting state, reward and a boolean indicating
        whether the termination criterion is reached or not.
        The boolean is_model_dynamic indicates whether the temporal transition is applied
        to the state vector or not.
        """
        d = self.transition_probability_distribution(s, s.time, a)
        p_p = categorical_sample(d, self.np_random)
        if is_model_dynamic:
            s_p = State(p_p, s.time + self.tau)
        else:
            s_p = State(p_p, s.time)
        r = self.instant_reward(s, s.time, a, s_p)
        done = self.is_terminal(s_p)
        return s_p, r, done

    def instant_reward(self, s, t, a, s_p):
        """
        Return the instant reward for transition s, t, a, s_p
        """
        newrow, newcol = self.to_m(s_p.index)
        newletter = self.desc[newrow, newcol]
        if newletter == b'G':
            return +1.0
        elif newletter == b'H':
            return -1.0
        else:
            return 0.0

    def expected_reward(self, s, t, a):
        """
        Return the expected reward function at s, t, a
        """
        R = 0.0
        d = self.transition_probability_distribution(s, t, a)
        for i in range(len(d)):
            s_p = State(i, s.time + self.tau)
            r_i = self.instant_reward(s, t, a, s_p)
            R += r_i * d[i]
        return R

    def is_terminal(self, s):
        """
        Return True if the input state is terminal.
        """
        row, col = self.to_m(s.index)
        letter = self.desc[row, col]
        done = bytes(letter) in b'GH'
        if s.time + self.tau >= self.nT:  # Timeout
            done = True
        return done

    def step(self, a):
        s, r, done = self.transition(self.state, a, True)
        self.state = s
        self.lastaction = a
        return (s, r, done, {})

    def render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.state.index // self.ncol, self.state.index % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = colorize.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile
