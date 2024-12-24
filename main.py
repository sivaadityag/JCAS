import sys
import matplotlib.pyplot as plt

import numpy as np

total_rounds = 50
packet_length = 10  # Constant for all rounds

np.random.seed(np.random.randint(total_rounds))

# Preliminaries

n_states = 2
p1 = 0.1
p2 = 0.9

r = 0.5


T = np.array([[r, 1 - r], [r, 1 - r]], dtype=float)

states = [0, 1]
rates = [0.6, 0.4]

# Transmitter observation

errors = np.array([packet_length * p1, packet_length * p2],
                  dtype=int)

likelihood_prob = np.zeros(n_states, dtype=float)

prior_prob = np.array([0.5, 0.5], dtype=float)
post_prob = np.zeros(n_states, dtype=float)

q = np.zeros([n_states], dtype=float)

# Stationary distribution is always equally likely
observed_state = np.random.choice(states, total_rounds, p=[r, 1-r])
map_state = np.array([], dtype=int)

RATE_g = np.array([], dtype=int)
RATE_m = np.array([], dtype=int)

for r in range(total_rounds):

    # q = np.zeros([n_states], dtype=float)

    channel_error = errors[observed_state[r]]
    likelihood_prob[observed_state[r]] = (channel_error / packet_length)

    # likelihood_prob[1 - observed_state[r]] = 1 - likelihood_prob[observed_state[r]]

    # Normalizing factor
    norm = 0
    for s in range(n_states):
        norm += likelihood_prob[s] * prior_prob[s]

    # q's are posterior of current observed state without memory

    q[observed_state[r]] = (likelihood_prob[observed_state[r]] * prior_prob[observed_state[r]]) / norm
    q[1 - observed_state[r]] = 1 - q[observed_state[r]]

    prior_prob = np.matmul(T.transpose(), q)

    # prior_prob[observed_state[r]] = T[observed_state[r]][observed_state[r]]*q[observed_state[r]] + T[observed_state[r]][1 - observed_state[r]]*q[1-observed_state[r]]

    # post_prob[observed_state[r]] = q[observed_state[r]]

    # MAP rule:
    # Normalizing factor

    norm_map = 0
    for s in range(n_states):
        norm_map += likelihood_prob[s] * 0.5

    post_prob[observed_state[r]] = (likelihood_prob[observed_state[r]] * 0.5) / norm_map
    post_prob[1 - observed_state[r]] = 1 - post_prob[observed_state[r]]

    # map_state = np.argmax(post_prob)
    map_state = np.argmax(q)
    RATE_m = np.append(RATE_m, rates[map_state])

    # GREEDY strategy:

    if observed_state[r] == 0 and (q[observed_state[r]] * rates[observed_state[r]] > rates[1 - observed_state[r]]):
        RATE_g = np.append(RATE_g, rates[observed_state[r]])
    else:
        RATE_g = np.append(RATE_g, rates[1 - observed_state[r]])

fig, ax = plt.subplots(1)

plt.plot(range(total_rounds), RATE_g, 'b*', linewidth=0.5, label="GREEDY")
plt.plot(range(total_rounds), RATE_m, 'r-', label="MAP")
# ax.scatter(range(total_rounds), RATE_g, marker='x', color='b', linewidth=0.3, label="GREEDY")
# ax.scatter(range(total_rounds), RATE_m, marker='o', color='r', linewidth=0.3, label="MAP")
plt.xlabel('TX rounds')
plt.ylabel('Rates')
plt.legend(loc="lower right")
plt.ylim(0.2, 0.7)
plt.show()
print(np.mean(RATE_g) >= np.mean(RATE_m), prior_prob, sum(prior_prob), sum(post_prob), sum(q))

# for i in range(total_rounds):
#     print(str(i) + " " + str(RATE_m[i]) + str("\\\\"))
