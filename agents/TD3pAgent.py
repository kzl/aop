import numpy as np
import torch
import copy

from agents.POLOAgent import POLOAgent
import utils.traj as traj
import pols.TD3 as TD3
from utils.ReplayBuffer_TD3 import ReplayBuffer

class TD3pAgent(POLOAgent):
    """
    An agent extending upon POLO that uses an independently trained TD3 policy
    as a prior to MPC.
    """

    def __init__(self, params):
        super(TD3pAgent, self).__init__(params)

        # TD3 module contains actor and critic
        self.TD3 = TD3.TD3(
            self.N, self.M, 1, 
            self.params['p-td3']['hidden_sizes'],
            device=self.device
        )

        self.td3_buf = ReplayBuffer(self.params['p-td3']['buf_size'])

        # Logging (store cum_rew, cum_emp_rew)
        self.hist['pols'] = np.zeros((self.T, 2))

        self.has_pol = True

        self.pol_cache = ()

    def get_action(self):
        """
        TD3pAgent generates a planned trajectory using the TD3-learned policy
        and then optimizes it via MPC.
        """

        # Run a rollout using the policy starting from the current state
        infos = self.get_traj_info()

        self.hist['pols'][self.time] = infos[3:5]
        self.pol_cache = (infos[0], infos[2])

        # Generate trajectory via MPC with the prior actions as a prior
        action = super(TD3pAgent, self).get_action(prior=infos[1])

        return action

    def print_logs(self):
        """
        TD3-specific logging information.
        """
        bi, ei = super(TD3pAgent, self).print_logs()

        self.print('TD3 metrics', mode='head')

        self.print('policy traj rew', 
            self.hist['pols'][self.time-1][0])
        self.print('policy traj emp rew', 
            self.hist['pols'][self.time-1][1])

        return bi, ei

    def do_updates(self):
        """
        Use the TD3 module to update the policy.
        """
        super(TD3pAgent, self).do_updates()

        if self.time % self.params['p-td3']['update_every'] == 0:
            if len(self.hist['plan'][self.time-1]) > 0:
                self.TD3.train(self.td3_buf, self.params['p-td3']['grad_steps'])

    def use_paths(self, paths):
        """
        After MPC generates many tajectories for planning, add them all to the
        replay buffer.
        """
        super(TD3pAgent, self).use_paths(paths)

        num_rollouts = len(paths)
        for i in range(num_rollouts):
            ps = self.prev_obs
            for t in range(paths[i][0].shape[0]):
                s = paths[i][0][t]
                a = paths[i][1][t]
                r = paths[i][2][t]

                # Add to buffer in TD3 replay buffer style
                self.td3_buf.add((ps, s, a, r, 0))
                ps = s

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
            pt=(self.TD3, 0), terminal=self.val_ens,
            tvel=self.tvel
        )

        return infos

    def test_policy(self):
        """
        Run the TD3 selection_action mechanism.
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
