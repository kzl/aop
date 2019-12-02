import numpy as np
import torch
import copy
import time

from agents.Agent import Agent
import utils.traj as traj
import pols.TD3 as TD3
from utils.ReplayBuffer_TD3 import ReplayBuffer

class TD3Agent(Agent):
    """
    An agent for running online TD3. The TD3 implementation itself is taken
    from the original authors, Fujimoto et. al. (see pols.TD3).
    """

    def __init__(self, params):
        super(TD3Agent, self).__init__(params)
        self.H = self.params['td3']['H']

        # TD3 module contains actor and critic
        self.TD3 = TD3.TD3(
            self.N, self.M, 1,
            hs=self.params['td3']['hidden_sizes'],
            device=self.device
        )

        self.td3_buf = ReplayBuffer(self.params['td3']['buf_size'])

        # Logging (store cum_rew, cum_emp_rew)
        self.hist['pols'] = np.zeros((self.T, 2))

    def get_action(self):
        """
        TD3 runs a deterministic policy.
        """

        # Logging policy information
        infos = self.get_traj_info()
        cum_rew, cum_emp_rew = infos[3:5]
        self.hist['pols'][self.time] = np.array([cum_rew, cum_emp_rew])

        # Select action
        act = self.TD3.select_action(self.prev_obs)

        return act

    def do_updates(self):
        """
        Update TD3 in online fashion by adding to the replay buffer in batch.
        """
        if self.time % self.params['td3']['update_every'] == 0 \
            or self.time == 1:

            plan_time = 0
            for iter_ind in range(self.params['td3']['num_iter']):
                check = time.time()
                rollouts = self.sample_rollouts(
                    self.H, self.params['td3']['num_rollouts']
                )
                plan_time += time.time() - check

                rews = np.zeros(len(rollouts))
                for i in range(len(rollouts)):
                    pobs = self.prev_obs
                    for k in range(self.H):
                        obs = rollouts[i][0][k]
                        act = rollouts[i][1][k]
                        rew = rollouts[i][2][k]
                        rews[i] += rew

                        self.td3_buf.add((pobs, obs, act, rew, 0))
                        pobs = obs

                self.TD3.train(self.td3_buf, self.params['td3']['grad_steps'])

            # Time spent generating rollouts should be considered planning time
            self.hist['plan_time'][self.time-1] += plan_time
            self.hist['update_time'][self.time-1] -= plan_time

    def sample_rollouts(self, H, num_rollouts):
        """
        Sample many rollouts from the policy in a large batch.
        """
        env_state = self.env.sim.get_state() if self.mujoco else None

        rollouts = traj.generate_trajectories(
            num_rollouts, self.env,
            env_state, self.prev_obs,
            mujoco=self.mujoco, perturb=self.perturb,
            H=H, gamma=self.gamma, act_mode='deter',
            pt=(self.TD3, self.params['td3']['act_std']),
            terminal=None, tvel=self.tvel,
            num_cpu=self.params['td3']['num_cpu']
        )

        return rollouts

    def get_traj_info(self):
        """
        Run the policy for a full trajectory (for logging purposes).
        """
        env_state = self.env.sim.get_state() if self.mujoco else None

        infos = traj.eval_traj(
            copy.deepcopy(self.env),
            env_state, self.prev_obs,
            mujoco=self.mujoco, perturb=self.perturb,
            H=self.H, gamma=self.gamma, act_mode='deter',
            pt=(self.TD3, 0), tvel=self.tvel
        )

        return infos

    def test_policy(self):
        """
        Run the TD3 action selection mechanism.
        """
        env = copy.deepcopy(self.env)
        obs = env.reset()

        if self.tvel is not None:
            env.set_target_vel(self.tvel)
            obs = env._get_obs()

        env_state = env.sim.get_state() if self.mujoco else None
        infos = traj.eval_traj(
            env, env_state, obs,
            mujoco=self.mujoco, perturb=self.perturb,
            H=self.eval_len, gamma=1, act_mode='deter',
            pt=(self.TD3, 0), tvel=self.tvel
        )

        self.hist['pol_test'][self.time] = infos[3]
