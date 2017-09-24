# Solving the Multi-armed Bandit Problem

The multi-armed bandit problem is as follows: we have _n_ rigged slot machines bandits each with their own probability distribution of success (which are inaccessible to you). Pulling the lever of any machine gives us either a reward of +1 (success) or -1 (failure). Our goal is to play many episodes of this game with the goal of *maximizing* our total accumulated rewards. 

In this code, we deploy an epsilon-greedy agent to play the multi-armed bandit game for a fixed number of episodes using a policy-based optimization method of policy gradient updates on a feed-forward neural network approximation of the policy. As a result of playing the game, we learn the optimal policy which in this case is the best bandit (slot machine with the highest probability of success).

We also provide appendices for the: 
* Epsilon-greedy agent (Appendix A)
* Policy gradients (Appendix B)

### Usage:

In the example provided, we train on 1000 experiments with 2000 episodes in each experiment. The default exploring parameter is `epsilon = 0.1` and 10 bandits are intialized with success probabilities of `{0.10, 0.50, 0.60, 0.80, 0.10, 0.25, 0.60, 0.45, 0.75, 0.65}`. To run the code, use 

> python multiarmed_bandit.py

The optimal policy should select bandit #4 as the "best" bandit minima on average, with bandit #9 as a close second.

### Libraries required:

* numpy

## Appendix A: The epsilon-greedy agent

The *epsilon-greedy agent* is an agent which at decision time either selects a greedy action with _1-epsilon_ probability, or explores the entire action space with the remaining _epsilon_ probability. Taking a greedy action means selecting the action with the highest expected reward.

Note that a mixture of exploration and greed is a quintessential aspect of reinforcement learning. Exploration is necessary for finding the optimal policy as the agent needs to understand the environment it is in, and to learn routes leaing to high long-term rewards. Although greed is not necessarily needed for finding the optimal policy, a purely exploratory agent does not utilize its experience to strategize for its future moves -- therefore no actual learning is involved without a greed mechanism in play. 