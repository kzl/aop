import numpy as np
import torch
import copy

from agents.VPGAgent import VPGAgent

class PPOAgent(VPGAgent):
    """
    Policy gradient-based agent using Proximal Policy Optimization (PPO) for
    policy updates.
    """

    def __init__(self, params):
        super(PPOAgent, self).__init__(params)
        self.eps = self.params['ppo']['eps']

    def update_pol(self, rollouts, orig_logprob=None):
        """
        Updating the policy is mostly the same, but must keep the original log
        probabilities from the start of the training for a timestep in order to
        calculate the PPO loss.
        """
        H = rollouts[0][0].shape[0]

        obs = np.zeros((len(rollouts), self.H, self.N))
        acts = np.zeros((len(rollouts), self.H, self.M))
        for i in range(len(rollouts)):
            for k in range(self.H):
                obs[i,k] = rollouts[i][0][k]
                acts[i,k] = rollouts[i][1][k]

        obs = torch.tensor(obs, dtype=self.dtype)
        acts = torch.tensor(acts, dtype=self.dtype)

        orig_logprob = self.get_logprob(self.pol, self.log_std, obs, acts)
        orig_logprob = orig_logprob.detach()
        
        rews = super(PPOAgent, self).update_pol(
            rollouts, orig_logprob=orig_logprob)

        return rews

    def get_pol_loss(self, logprob, advs, orig_logprob):
        """
        Calculate the policy loss in PPO fashion. This will be negated for the
        gradient ascent in the parent class (i.e. this function should return
        the value to be maximized).
        """
        prob_rat = torch.exp(logprob - orig_logprob)
        advs_rat = prob_rat * advs

        # PPO-Clip procedure
        prob_clip = torch.clamp(prob_rat, 1-self.eps, 1+self.eps)
        advs_clip = prob_clip * advs

        min_advs = torch.min(advs_rat, advs_clip)

        return torch.mean(min_advs)

    def continue_updates(self, logprob, orig_logprob=None):
        """
        PPO can terminate updates when the target KL has reached a certain
        threshold.
        """

        return True
