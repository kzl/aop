import numpy as np
import torch
import copy

from agents.MPCAgent import MPCAgent
from utils.ReplayBuffer import ReplayBuffer
from models.Ensemble import Ensemble

class POLOAgent(MPCAgent):
    """
    MPC-based agent that uses the Plan Online, Learn Offline (POLO) framework
    (Lowrey et. al. 2018) for trajectory optimization.
    """

    def __init__(self, params):
        super(POLOAgent, self).__init__(params)
        self.H_backup = self.params['polo']['H_backup']

        # Create ensemble of value functions
        model_params = params['polo']['ens_params']['model_params']
        model_params['input_size'] = self.N
        model_params['output_size'] = 1

        params['polo']['ens_params']['dtype'] = self.dtype
        params['polo']['ens_params']['device'] = self.device

        self.val_ens = Ensemble(self.params['polo']['ens_params'])

        # Learn from replay buffer
        self.polo_buf = ReplayBuffer(
            self.N, self.M, self.params['polo']['buf_size'])

        # Value (from forward), value mean, value std
        self.hist['vals'] = np.zeros((self.T, 3))

    def get_action(self, prior=None):
        """
        POLO selects action based on MPC optimization with an optimistic
        terminal value function.
        """
        self.val_ens.eval()

        # Get value of current state
        s = torch.tensor(self.prev_obs, dtype=self.dtype)
        s = s.to(device=self.device)
        current_val = self.val_ens.forward(s)[0]
        current_val = torch.squeeze(current_val, -1)
        current_val = current_val.detach().cpu().numpy()

        # Get prediction of every function in ensemble
        preds = self.val_ens.get_preds_np(self.prev_obs)

        # Log information from value function
        self.hist['vals'][self.time] = \
            np.array([current_val, np.mean(preds), np.std(preds)])

        # Run MPC to get action
        act = super(POLOAgent, self).get_action(
            terminal=self.val_ens, prior=prior)

        return act

    def action_taken(self, prev_obs, obs, rew, done, ifo):
        """
        Update buffer for value function learning.
        """
        self.polo_buf.update(prev_obs, obs, rew, done)

    def do_updates(self):
        """
        POLO learns a value function from its past true history of interactions
        with the environment.
        """
        super(POLOAgent, self).do_updates()
        if self.time % self.params['polo']['update_freq'] == 0:
            self.val_ens.update_from_buf(
                self.polo_buf, self.params['polo']['grad_steps'],
                self.params['polo']['batch_size'],
                self.params['polo']['H_backup'], self.gamma
            )

    def print_logs(self):
        """
        POLO-specific logging information.
        """
        bi, ei = super(POLOAgent, self).print_logs()

        self.print('POLO metrics', mode='head')

        self.print('current state val',
            self.hist['vals'][self.time-1][0])
        self.print('current state std',
            self.hist['vals'][self.time-1][2])

        return bi, ei
