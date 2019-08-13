import numpy as np

err = []
nS = 3
p = np.random.random((nS,))
p /= np.sum(p)
#p = np.array([0.1, 0.5, 0.4])
n_samples = int(1e5)
n_known = 100

for i in range(n_samples):  # learn n_samples models
    p_i = np.zeros((nS,))

    out = np.random.choice(nS, size=(n_known,), p=p)

    for o in out:
        p_i[o] += 1
    p_i = p_i / float(n_known)

    e_i = np.sum(np.abs(p - p_i))
    err.append(e_i)

print('max  :', max(err))
print('mean :', np.mean(err))
