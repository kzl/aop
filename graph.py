import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import os
import pickle
import sys

"""
This is an example script to graph the rewards achieved over the agents'
lifetimes. Note that these are *per-timestep* rewards. Smoothing helps to show
the average reward around that timestep, making it look more like an episode --
it is especially important to use large smoothing (>=500) on the mazes to show
an accurate measure of performance.

It can be easily modified to plot other values by substituting the value for
metric; see agents/Agent.py to see what is being stored by experiments.
"""

def main():
    if len(sys.argv) < 3:
        print('Command format: python graph.py <dir_name> <lifetime_len>')
        return

    directory, T = sys.argv[1], int(sys.argv[2])
    metric = 'rew'

    # Gather agents

    agents, agent_files = {}, next(os.walk(directory))[1]
    agents_found = 0
    for file_name in agent_files:
        info = (directory, file_name, T)

        try:
            f = open('%s/%s/checkpoints/model_%d.pkl' % info, 'rb')
            agent = pickle.load(f)
            f.close()
        except Exception as e:
            continue

        algo = agents.algo
        if algo in agents:
            agents[algo].append(agent)
        else:
            agents[algo] = [agent]

        agents_found += 1

    if agents_found == 0:
        info = (directory, T)
        print('Error: could not find any agents in %s for time %d.' % info)
        return

    # Calculate running means and stds

    rew_mean = {algo: np.zeros(T) for algo in agents}
    rew_std = {algo: np.zeros(T) for algo in agents}
    smoothing = 250 # set to 0 to remove smoothing

    for algo in agents:
        for t in range(T):
            rews_t = []
            bi, ei = max(0, t-smoothing), min(T, t+smoothing+1)
            for agent in agents[algo]:
                rews_t.append(np.mean(agent.hist[metric][bi:ei]))
            rew_mean[algo][t] = np.mean(rews_t)
            rew_std[algo][t] = np.std(rews_t)

    # Plot means with one standard deviation

    plt.title('Rewards achieved over agent lifetime')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')

    for algo in agents:
        plt.plot(rew_mean[algo], label=algo, linewidth=2)
        plt.fill_between(
            np.linspace(0, T-1, T),
            rew_mean[algo]-rew_std[algo],
            rew_mean[algo]+rew_std[algo],
            alpha=0.3
        )

    plt.legend(loc='best')

    plt.savefig(directory + '/rewards.png')

if __name__ == '__main__':
    main()
