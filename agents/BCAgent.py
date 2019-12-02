import numpy as np
import torch
import copy

from agents.POLOAgent import POLOAgent
from models.MLP import MLP
from utils.ReplayBuffer import ReplayBuffer
import utils.traj as traj

class BCAgent(POLOAgent):
    """
    An agent extending upon POLO that uses behavior cloning on the planner
    predicted actions as a prior to MPC.
    """

    def __init__(self, params):
        super(BCAgent, self).__init__(params)

        # Initialize policy network
        pol_params = self.params['p-bc']['pol_params']
        pol_params['input_size'] = self.N
        pol_params['output_size'] = self.M
        if 'final_activation' not in pol_params:
            pol_params['final_activation'] = torch.tanh

        self.pol = MLP(pol_params)

        # Create policy optimizer
        ppar = self.params['p-bc']['pol_optim']
        self.pol_optim = torch.optim.Adam(
            self.pol.parameters(), lr=ppar['lr'],
            weight_decay=ppar['reg']
        )

        # Use a replay buffer that will save planner actions
        self.pol_buf = ReplayBuffer(
            self.N, self.M, self.params['p-bc']['buf_size']
        )

        # Logging (store cum_rew, cum_emp_rew)
        self.hist['pols'] = np.zeros((self.T, 2))

        self.has_pol = True

        self.pol_cache = ()
        
    def get_action(self):
        """
        BCAgent generates a planned trajectory using the behavior-cloned policy
        and then optimizes it via MPC.
        """
        self.pol.eval()

        # Run a rollout using the policy starting from the current state
        infos = self.get_traj_info()

        self.hist['pols'][self.time] = infos[3:5]
        self.pol_cache = (infos[0], infos[2])

        self.prior_actions = infos[1]

        # Generate trajectory via MPC with the prior actions as a prior
        action = super(BCAgent, self).get_action(prior=self.prior_actions)

        # Add final planning trajectory to BC buffer
        fin_states, fin_rews = self.cache[2], self.cache[3]
        fin_states = np.concatenate(([self.prev_obs], fin_states[1:]))
        pb_pct = self.params['p-bc']['pb_pct']
        pb_len = int(pb_pct * fin_states.shape[0])
        for t in range(pb_len):
            self.pol_buf.update(
                fin_states[t], fin_states[t+1],
                fin_rews[t], self.planned_actions[t], False
            )

        return action

    def do_updates(self):
        """
        Learn from the saved buffer of planned actions.
        """
        super(BCAgent, self).do_updates()

        if self.time % self.params['p-bc']['update_freq'] == 0:
            self.update_pol()

    def update_pol(self):
        """
        Update the policy via BC on the planner actions.
        """
        self.pol.train()

        params = self.params['p-bc']

        # Generate batches for training
        size = min(self.pol_buf.size, self.pol_buf.total_in)
        num_inds = params['batch_size'] * params['grad_steps']
        inds = np.random.randint(0, size, size=num_inds)

        states = self.pol_buf.buffer['s'][inds]
        acts = self.pol_buf.buffer['a'][inds]

        states = torch.tensor(states, dtype=self.dtype)
        actions = torch.tensor(acts, dtype=self.dtype)

        for i in range(params['grad_steps']):
            bi, ei = i*params['batch_size'], (i+1)*params['batch_size']

            # Train based on L2 distance between actions and predictions
            preds = self.pol.forward(states[bi:ei])
            preds = torch.squeeze(preds, dim=-1)
            targets = torch.squeeze(actions[bi:ei], dim=-1)

            loss = torch.nn.functional.mse_loss(preds, targets)

            self.pol_optim.zero_grad()
            loss.backward()
            self.pol_optim.step()

    def get_traj_info(self):
        """
        Run the policy for a full trajectory and return details about the
        trajectory.
        """
        env_state = self.env.sim.get_state() if self.mujoco else None

        infos = traj.eval_traj(
            copy.deepcopy(self.env),
            env_state, self.prev_obs,
            mujoco=self.mujoco, perturb=self.perturb,
            H=self.H, gamma=self.gamma, act_mode='deter',
            pt=(self.pol, 0), terminal=self.val_ens,
            tvel=self.tvel
        )

        return infos

    def print_logs(self):
        """
        BC-specific logging information.
        """
        bi, ei = super(BCAgent, self).print_logs()

        self.print('BC metrics', mode='head')

        self.print('policy traj rew', 
            self.hist['pols'][self.time-1][0])
        self.print('policy traj emp rew', 
            self.hist['pols'][self.time-1][1])

        return bi, ei

    def test_policy(self):
        """
        Run the BC action selection mechanism.
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
            pt=(self.pol, 0), tvel=self.tvel
        )

        self.hist['pol_test'][self.time] = infos[3]
