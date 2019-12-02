import numpy as np
import torch
import copy
import time

from agents.Agent import Agent
import utils.traj as traj
from models.MLP import MLP

class VPGAgent(Agent):
    """
    An agent running online policy gradient. Calling VPGAgent itself uses
    REINFORCE, but can be subclassed for other policy gradient class algorithms.
    """

    def __init__(self, params):
        super(VPGAgent, self).__init__(params)
        self.H = self.params['pg']['H']
        self.lam = self.params['pg']['lam']

        # Initialize policy network
        pol_params = self.params['pg']['pol_params']
        pol_params['input_size'] = self.N
        pol_params['output_size'] = self.M
        if 'final_activation' not in pol_params:
            pol_params['final_activation'] = torch.tanh

        self.pol = MLP(pol_params)

        # Std's are not dependent on state
        init_log_std = -0.8 * torch.ones(self.M) # ~0.45
        self.log_std = torch.nn.Parameter(init_log_std, requires_grad=True)

        # Create policy optimizer
        ppar = self.params['pg']['pol_optim']
        self.pol_params = list(self.pol.parameters()) + [self.log_std]
        self.pol_optim = torch.optim.Adam(
            self.pol_params, lr=ppar['lr'],
            weight_decay=ppar['reg']
        )

        # Create value function and optimizer
        val_params = self.params['pg']['val_params']
        val_params['input_size'] = self.N
        val_params['output_size'] = 1

        self.val = MLP(val_params)

        vpar = self.params['pg']['val_optim']
        self.val_optim = torch.optim.Adam(
            self.val.parameters(), lr=vpar['lr'],
            weight_decay=vpar['reg']
        )

        # Logging
        self.hist['ent'] = np.zeros(self.T)

    def get_dist(self, s):
        """
        Create a pytorch normal distribution from
        the policy network for state s.
        """
        s = torch.tensor(s, dtype=self.dtype)
        mu = self.pol.forward(s)
        std = self.log_std.exp()

        return torch.distributions.Normal(mu, std)

    def get_ent(self):
        """
        Return the current entropy (multivariate Gaussian).
        """
        std = self.log_std.exp()
        tpe = 2 * np.pi * np.e
        return .5 * torch.log(tpe * torch.prod(std))

    def get_action(self):
        """
        Gets action by running policy.
        """
        self.pol.eval()

        if self.params['pg']['run_deterministic']:
            x = torch.tensor(self.prev_obs, dtype=self.dtype)
            act = self.pol.forward(x).detach().cpu().numpy()
        else:
            act = sample_pol(self.pol, self.log_std, self.prev_obs)

        act = np.clip(act, 
            self.params['env']['min_act'], self.params['env']['max_act'])

        self.hist['ent'][self.time] = self.get_ent().detach().cpu().numpy()

        return act

    def do_updates(self):
        """
        Performs actor and critic updates.
        """
        if self.time % self.params['pg']['update_every'] == 0 or self.time == 1:
            plan_time = 0
            H, num_rollouts = self.H, self.params['pg']['num_rollouts']
            for i in range(self.params['pg']['num_iter']):
                # Sample rollouts using ground truth model
                check = time.time()
                rollouts = self.sample_rollouts(H, num_rollouts)
                plan_time += time.time() - check
                
                # Performs value updates alongside advantage calculation
                rews = self.update_pol(rollouts)

            # Time spent generating rollouts should be considered planning time
            self.hist['plan_time'][self.time-1] += plan_time
            self.hist['update_time'][self.time-1] -= plan_time

    def sample_rollouts(self, H, num_rollouts):
        """
        Use traj module to sample rollouts using the policy.
        """
        env_state = self.env.sim.get_state() if self.mujoco else None

        self.pol.eval()
        rollouts = traj.generate_trajectories(
            num_rollouts,
            self.env, env_state, self.prev_obs,
            mujoco=self.mujoco, perturb=self.perturb,
            H=self.H, gamma=self.gamma,
            act_mode='gauss',
            pt=(sample_pol, self.pol, self.log_std),
            terminal=None,
            tvel=self.tvel,
            num_cpu=self.params['pg']['num_cpu']
        )

        return rollouts

    def update_val(self, obs, targets):
        """
        Update value function with MSE loss.
        """
        preds = self.val.forward(obs)
        preds = torch.squeeze(preds, dim=-1)

        loss = torch.nn.functional.mse_loss(targets, preds)

        self.val_optim.zero_grad()
        loss.backward(retain_graph=True)
        self.val_optim.step()

        return loss.item()

    def calc_advs(self, obs, rews, update_vals=True):
        """
        Calculate advantages for use of updating the policy (and updating value
        function). Can either use rewards-to-go or GAE.
        """
        num_rollouts, H = obs.shape[:2]

        self.val.eval()

        if not self.params['pg']['use_gae']:
            # Calculate terminal values
            fin_obs = obs[:,-1]
            fin_vals = self.val.forward(fin_obs)
            fin_vals = torch.squeeze(fin_vals, dim=-1)

            # Calculate rewards-to-go
            rtg = torch.zeros((num_rollouts, H))
            for k in reversed(range(H)):
                if k < H-1:
                    rtg[:,k] += self.gamma * rtg[:,k+1]
                else:
                    rtg[:,k] += self.gamma * fin_vals
                rtg[:,k] += rews[:,k]

            if update_vals:
                self.val.train()
                self.update_val(obs, rtg)

            # Normalize advantages for policy gradient
            for k in range(H):
                rtg[:,k] -= torch.mean(rtg[:,k])

            return rtg

        # Generalized Advantage Estimation (GAE)
        prev_obs = torch.tensor(self.prev_obs, dtype=self.dtype)
        orig_val = self.val.forward(prev_obs)
        vals = torch.squeeze(self.val.forward(obs), dim=-1)
        
        deltas = torch.zeros(rews.shape)
        advs = torch.zeros((num_rollouts, H))

        lg = self.lam * self.gamma
        for k in reversed(range(H)):
            prev_vals = vals[:,k-1] if k > 0 else orig_val
            deltas[:,k] = self.gamma*vals[:,k]+ rews[:,k] - prev_vals

            if k == H-1:
                advs[:,k] = deltas[:,k]
            else:
                advs[:,k] = lg * advs[:,k+1] + deltas[:,k]

        advs = advs.detach()

        # Optionally, also update the value functions
        if update_vals:
            self.val.train()

            # It is reasonable to train on advs or deltas
            dvals = advs

            # Have to perform trick to match deltas with prev vals
            fvals = torch.stack([orig_val for _ in range(vals.shape[0])], dim=0)
            rets = torch.cat(
                [fvals + dvals[:,:1], vals[:,:-1] + dvals[:,1:]], dim=-1)
            fobs = torch.unsqueeze(prev_obs, dim=0)
            fobs = torch.stack([fobs for _ in range(vals.shape[0])], dim=0)
            obs = torch.cat([fobs, obs[:,:-1]], dim=1)

            self.update_val(obs, rets)

        # Normalize advantages for policy gradient
        advs -= torch.mean(advs)
        advs /= 1e-3 + torch.std(advs)

        return advs

    def get_pol_loss(self, logprob, advs, orig_logprob=None):
        """
        For REINFORCE, the policy loss is thelogprobs times the advatanges. It
        is important that the logprobs carry the gradient so that we can
        backpropagate through them in the policy update.
        """
        return torch.mean(logprob * advs)

    def get_logprob(self, pol, log_std, obs, acts):
        """
        Get log probabilities for the actions, keeping the gradients.
        """
        num_rollouts, H = obs.shape[0:2]

        pol.train()
    
        dist = self.get_dist(obs)
        logprob = dist.log_prob(acts).sum(-1)

        return logprob

    def update_pol(self, rollouts, orig_logprob=None):
        """
        Update the policy on the on-policy rollouts.
        """
        H = rollouts[0][0].shape[0]

        self.pol.train()

        obs = np.zeros((len(rollouts), self.H, self.N))
        acts = np.zeros((len(rollouts), self.H, self.M))
        rews = torch.zeros((len(rollouts), self.H))
        for i in range(len(rollouts)):
            for k in range(self.H):
                obs[i,k] = rollouts[i][0][k]
                acts[i,k] = rollouts[i][1][k]
                rews[i,k] = rollouts[i][2][k]

        obs = torch.tensor(obs, dtype=self.dtype)
        acts = torch.tensor(acts, dtype=self.dtype)

        # Perform updates for multiple steps on the value function
        if self.params['pg']['use_gae']:
            for _ in range(self.params['pg']['val_steps']):
                advs = self.calc_advs(obs, rews, update_vals=True)
        else:
            advs = self.calc_advs(obs, rews, update_vals=False)

        # Perform updates for multiple epochs on the policy
        bsize = self.params['pg']['batch_size']
        for _ in range(self.params['pg']['pol_steps']):
            inds = np.random.permutation(len(rollouts))

            binds = inds[:bsize]
            bobs, bacts = obs[binds], acts[binds]
            brews, badvs = rews[binds], advs[binds]

            if orig_logprob is not None:
                bprobs = orig_logprob[binds]
            else:
                bprobs = None

            # Get a logprob that has gradients
            logprob = self.get_logprob(
                self.pol, self.log_std, bobs, bacts
            )
            if not self.continue_updates(logprob, bprobs):
                break

            # Compute policy loss (i.e. gradient ascent)
            J = -self.get_pol_loss(
                logprob, badvs, orig_logprob=bprobs
            )

            # Apply entropy bonus
            ent_coef = self.params['pg']['pol_optim']['ent_temp']
            if ent_coef != 0:
                J -= ent_coef * self.get_ent()

            self.pol_optim.zero_grad()
            torch.nn.utils.clip_grad_norm_(
                self.pol.parameters(),
                self.params['pg']['grad_clip'])
            J.backward()
            self.pol_optim.step()

            # Clamp stds to be within set bounds
            log_min = np.log(self.params['pg']['min_std'])
            log_min = torch.tensor(log_min, dtype=self.dtype)
            log_max = np.log(self.params['pg']['max_std'])
            log_max = torch.tensor(log_max, dtype=self.dtype)
            self.log_std.data = torch.clamp(
                self.log_std.data, log_min, log_max)

        return rews

    def continue_updates(self, logprob, orig_logprob=None):
        """
        Method for whether or not to continue updates.
        """
        return True

    def print_logs(self):
        """
        Policy gradient-specific logging information.
        """
        bi, ei = super(VPGAgent, self).print_logs()

        self.print('policy gradient metrics', mode='head')

        self.print('entropy avg',
            np.mean(self.hist['ent'][bi:ei]))
        self.print('sigma avg',
            np.mean(torch.exp(self.log_std).detach().cpu().numpy()))

        return bi, ei

def sample_pol(pol, log_std, s, min_act=-1, max_act=1):
    """
    Samples an action from a Gaussian policy.
    """
    M = log_std.shape[0]

    # Run policy network to get mu and std
    s = torch.tensor(s, dtype=pol.dtype)
    act = pol.forward(s)
    stds = torch.exp(log_std)

    # Apply noise to action
    noise = torch.tensor(np.random.normal(size=M), dtype=pol.dtype)
    noise *= stds
    act = torch.clamp(act+noise, -1, 1)

    return act.detach().cpu().numpy()
