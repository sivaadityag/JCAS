import sys
import matplotlib.pyplot as plt
from scipy.stats import binom
import numpy as np

n = 50
N = 5

# np.random.seed(np.random.randint(n))

# Preliminaries

n_states = 2
p1 = 0.09
p2 = 0.25
BSC_prob = np.array([p1, p2])
r = 0.95
T = np.array([[r, 1 - r], [r, 1 - r]], dtype=float)
states = [0, 1]
rates = np.array([0.3, 0.1])

error_prob = np.zeros(N + 1, dtype=float)

for i in range(N + 1):
    error_prob[i] = (binom.pmf(i, N, p1) * p1 + binom.pmf(i, N, p2) * p2) / (p1 + p2)

likelihood_prob = np.zeros(n_states, dtype=float)

prior_prob = np.array([0.5, 0.5], dtype=float)
post_prob = np.zeros(n_states, dtype=float)

q = np.zeros([n_states], dtype=float)

# Stationary distribution is always equally likely

observed_state = np.random.choice(states, n, p=[r, 1 - r])
# map_state = np.array([], dtype=int)

# RATE_g = np.array([rates[1]])  # GREEDY strategy
# RATE_m = np.array([rates[1]])  # MAP with uniform priors
# RATE_m_q = np.array([rates[1]])  # MAP with updated priors

RATE_g = np.array([])  # GREEDY strategy
RATE_m = np.array([])  # MAP with uniform priors
RATE_m_q = np.array([])  # MAP with updated priors

# Round increase

round_m = 0
round_m_q = 0
round_mq_vm = 0

# Estimation cost

cost_map = 0
cost_map_q = 0

for r in range(n):

    channel_error = np.random.choice(np.array(list(range(0, N + 1))), size=1, p=error_prob)[0]

    likelihood_prob[observed_state[r]] = binom.pmf(channel_error, N, BSC_prob[observed_state[r]])

    likelihood_prob[1 - observed_state[r]] = 1 - likelihood_prob[observed_state[r]]

    # Normalizing factor
    norm = 0
    for s in range(n_states):
        norm += likelihood_prob[s] * prior_prob[s]

    # q's are posterior of current observed state without memory

    q[observed_state[r]] = (likelihood_prob[observed_state[r]] * prior_prob[observed_state[r]]) / norm
    q[1 - observed_state[r]] = 1 - q[observed_state[r]]

    prior_prob = np.matmul(T.transpose(), q)

    # post_prob[observed_state[r]] = q[observed_state[r]]

    # MAP rule:
    # Normalizing factor

    norm_map = 0
    for s in range(n_states):
        norm_map += likelihood_prob[s] * 0.5

    post_prob[observed_state[r]] = (likelihood_prob[observed_state[r]] * 0.5) / norm_map
    post_prob[1 - observed_state[r]] = 1 - post_prob[observed_state[r]]

    # temp = likelihood_prob * 0.5
    # map_state = int(np.argmax(temp))

    map_state = int(np.argmax(post_prob))  # Basic MAP
    RATE_m = np.append(RATE_m, rates[map_state])

    map_q_state = int(np.argmax(q))  # Modified MAP
    RATE_m_q = np.append(RATE_m_q, rates[map_q_state])

    if map_state != int(observed_state[r]):
        cost_map += 1
    if map_q_state != int(observed_state[r]):
        cost_map_q += 1

    # GREEDY strategy:

    if prior_prob[0] * rates[0] > rates[1]:
        RATE_g = np.append(RATE_g, rates[0])
    else:
        RATE_g = np.append(RATE_g, rates[1])

    if RATE_g[r] > RATE_m[r]:
        round_m += 1

    if RATE_g[r] > RATE_m_q[r]:
        round_m_q += 1
    if RATE_m_q[r] > RATE_m[r]:
        round_mq_vm += 1

    # assert RATE_g[r] >= RATE_m[r]
fig, ax = plt.subplots(1)

ax.scatter(range(n), RATE_g, marker="v", color='b', linewidth=0.9, label="GREEDY")
ax.scatter(range(n), RATE_m, marker='^', color='r', linewidth=0.1, label="MAP")
ax.scatter(range(n), RATE_m_q, marker='o', color='g', linewidth=0.1, label="MAP_q")

# ax.scatter(range(n + 1), RATE_g, marker="v", color='b', linewidth=0.9, label="GREEDY")
# ax.scatter(range(n + 1), RATE_m, marker='^', color='r', linewidth=0.1, label="MAP")
# ax.scatter(range(n + 1), RATE_m_q, marker='o', color='g', linewidth=0.1, label="MAP_q")

plt.xlabel('TX rounds')
plt.ylabel('Rates')
plt.legend(loc="lower right")
plt.ylim(0.05, 0.4)
plt.show()

# for i in range(n):
#     print(str(i + 1) + " " + str(RATE_g[i]) + str("\\\\"))
# print("I'm done: GREEDY")
# for i in range(n):
#     print(str(i + 1) + " " + str(RATE_m_q[i]) + str("\\\\"))
# print("I'm done: MAP_q")
# for i in range(n):
#     print(str(i + 1) + " " + str(RATE_m[i]) + str("\\\\"))

print(np.mean(RATE_g) > np.mean(RATE_m), np.mean(RATE_g) > np.mean(RATE_m_q), prior_prob, sum(prior_prob),
      sum(post_prob), sum(q), cost_map / n, cost_map_q / n, np.mean(RATE_g), np.mean(RATE_m_q),
      np.mean(RATE_m), round_m / n, round_m_q / n, round_mq_vm / n)  # RATE_g == RATE_m_q)

# print(np.mean(RATE_g) > np.mean(RATE_m), prior_prob, sum(prior_prob),
#       sum(post_prob), sum(q), cost_map / n, cost_map_q / n, np.mean(RATE_g[1:]), np.mean(RATE_m))  # RATE_g == RATE_m_q)
