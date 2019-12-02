import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm

import copy
import multiprocessing as mp
import os
import pickle
import random
import time

from agents.Agent import Agent
import utils.traj as traj
from models.MLP import MLP

class MPCAgent(Agent):
	"""
	A vanilla MPC agent that optimizes trajectories using Model Path Predictive
	Integral (MPPI) control. Planning is done online and starts new planning by
	iteration off of the old plan.

	Will run AOP-style planning is self.has_aop is True at time of planning.
	"""

	def __init__(self, params):
		super(MPCAgent, self).__init__(params)
		self.H = self.params['mpc']['H']

		# Store planned trajectory, and copy for extending
		self.planned_actions = [np.zeros(self.M) for _ in range(self.H)]
		self.ghost_plan = [np.zeros(self.M) for _ in range(self.H)]

		# Logging history of agent/environment
		self.hist['plan'] = [[] for _ in range(self.T)]
		self.hist['init_plan'] = np.zeros((self.T, 2))
		self.hist['prior_des'] = []
		self.hist['H_des'] = np.zeros(self.T)
		self.hist['plan_belerr'] = [[] for _ in range(self.T)]

		# To allow for specific functions to be called
		self.has_pol = False
		self.has_aop = False

		# Optionally, we can only consider empirical reward
		self.use_terminal = self.params['mpc']['use_terminal']

		self.init_cache = ()

	def get_action(self, terminal=None, prior=None):
		"""
		Get the action as the first action of an iteratively improved planned
		trajectory. Optionally, use a terminal value function to help evaluate
		trajectories, or start with a specific prior trajectory.
		"""
		self.extend_plan(self.H)

		# Log initial trajectory
		env_state = self.env.sim.get_state() if self.mujoco else None
		terminal = terminal if self.use_terminal else None
		init_states, _, init_rews, cum_rew, emp_rew = \
			traj.eval_traj(
				copy.deepcopy(self.env),
				env_state, self.prev_obs,
				mujoco=self.mujoco, perturb=self.perturb,
				H=self.H, gamma=self.gamma, act_mode='fixed',
				pt=(np.array(self.planned_actions), 0),
				terminal=terminal, tvel=self.tvel
			)

		self.hist['init_plan'][self.time] = [cum_rew, emp_rew]

		self.init_cache = (init_states, init_rews, emp_rew)

		# Use the prior as the starting trajectory, if better
		if prior is not None:
			pol_val = self.hist['pols'][self.time][0]
			ratio = (cum_rew-pol_val) / abs(pol_val)
			if pol_val > cum_rew:
				self.hist['prior_des'].append('pol')
				self.planned_actions = []
				for i in range(prior.shape[0]):
					self.planned_actions.append(prior[i])

				# Store for calculations later
				init_states = self.pol_cache[0]
				init_rews = self.pol_cache[1]
			else:
				self.hist['prior_des'].append('plan')
		else:
			ratio = 1

		# Determine if we should use a lower planning horizon
		H = self.H
		if self.has_aop:
			# First, check std threshold
			std = self.hist['vals'][self.time][2]
			force_full_H = std > self.params['aop']['std_thres']

			# Pick Bellman error based on future values
			rews = init_rews
			states_t = torch.tensor(init_states, dtype=self.dtype)
			states_t = states_t.to(device=self.device)
			vals = self.val_ens.forward(states_t)
			vals = torch.squeeze(vals, dim=-1)
			vals = vals.detach().cpu().numpy()
			cval = self.hist['vals'][self.time][0]

			# Calculate the Bellman error from each state
			belerrs, run_rew = np.zeros(rews.shape[0]), 0
			for k in reversed(range(self.H)):
				run_rew = rews[k] + self.gamma * run_rew
				cv = vals[k-1] if k > 0 else cval
				belerrs[k] = abs(run_rew + vals[-1] - cv)

			if force_full_H:
				H = self.H
			else:
				# Use the first H where the Bellman error is too high
				H = 1
				BE_thres = self.params['aop']['bellman_thres']
				for k in reversed(range(self.H)):
					if belerrs[k] > BE_thres:
						H += k
						break

				# Get the value of the plan w.r.t. new H
				cum_rew = (self.gamma ** H) * vals[H-1]
				for t in range(H):
					cum_rew += rews[t]

		# Logging for AOP
		self.hist['H_des'][self.time] = H
		if self.has_aop:
			self.hist['plan_belerr'][self.time].append(belerrs)

		# Store best plan found by MPC
		best_plan_val = cum_rew
		best_plan = copy.deepcopy(self.planned_actions)
		old_plan_val = cum_rew

		# Run multiple iterations of trajectory optimization
		for i in range(self.params['mpc']['num_iter']):
			# Continue if the ratio threshold is met
			_continue = True
			if self.has_aop:
				thres = self.params['aop']['init_thres'] if i == 0 else \
						self.params['aop']['ratio_thres']
				rprob = random.random() < self.params['aop']['eps_plan']
				_continue = (ratio > thres) or rprob
			if not _continue:
				break

			num_rollouts = self.params['mpc']['num_rollouts']
			
			self.planned_actions = copy.deepcopy(best_plan)
			self.extend_plan(H)

			# Run MPC optimization
			self.update_plan(terminal, num_rollouts)

			self.finished_plan_update(i)

			# Update best plan found by MPC
			new_plan_val = self.hist['plan'][self.time][i][2]
			ratio = (new_plan_val - old_plan_val) / abs(old_plan_val)
			if new_plan_val > best_plan_val:
				best_plan_val = new_plan_val
				best_plan = copy.deepcopy(self.planned_actions)


			old_plan_val = new_plan_val

		self.planned_actions = copy.deepcopy(best_plan)

		# Measure final planning information metrics
		env_state = self.env.sim.get_state() if self.mujoco else None
		fin_states, _, fin_rews, fin_cum_rew, fin_emp_rew = \
			traj.eval_traj(
				copy.deepcopy(self.env),
				env_state, self.prev_obs,
				mujoco=self.mujoco, perturb=self.perturb,
				H=len(self.planned_actions),
				gamma=self.gamma, act_mode='fixed',
				pt=(np.array(self.planned_actions), 0),
				terminal=terminal, tvel=self.tvel
			)

		if len(self.cache) == 0:
			self.cache = ([], [])
		self.cache = (
			self.cache[0], self.cache[1], 
			fin_states, fin_rews
		)

		# Increment the trajectory by a timestep
		action = self.advance_plan()

		return action

	def update_plan(self, terminal=None, num_rollouts=None):
		"""
		Update the plan by generating noisy rollouts and then running MPPI.
		"""
		H = len(self.planned_actions)
		if num_rollouts is None:
			num_rollouts = self.params['mpc']['num_rollouts']

		# Generate and execute noisy rollouts
		plan = np.array(self.planned_actions)
		filter_coefs = self.params['mpc']['filter_coefs']
		env_state = self.env.sim.get_state() if self.mujoco else None
		paths = traj.generate_trajectories(
			num_rollouts, self.env,
			env_state, self.prev_obs,
			mujoco=self.mujoco, perturb=self.perturb,
			H=H, gamma=self.gamma, act_mode='fixed',
			pt=(plan, filter_coefs),
			terminal=None, tvel=self.tvel,
			num_cpu=self.params['mpc']['num_cpu']
		)

		num_rollouts = len(paths)

		# Evaluate rollouts with the terminal value function
		if terminal is not None:
			final_states = np.zeros((num_rollouts, self.N))
			for i in range(num_rollouts):
				final_states[i] = paths[i][0][-1]

			# Calculate terminal values
			fs = torch.tensor(final_states, dtype=self.dtype)
			terminal_vals = terminal.forward(fs).detach().cpu().numpy()
			
			# Append terminal values to cum_rew
			for i in range(num_rollouts):
				paths[i][3] += terminal_vals[i]

		self.use_paths(paths)

		plan_rews = np.zeros(num_rollouts)
		plan_rews_emp = np.zeros(num_rollouts)
		states = np.zeros((num_rollouts, H, self.N))
		actions = np.zeros((num_rollouts, H, self.M))

		for i in range(num_rollouts):
			plan_rews[i] = paths[i][3]
			plan_rews_emp[i] = paths[i][4]
			states[i] = paths[i][0]
			actions[i] = paths[i][1]

		if self.params['mpc']['temp'] is not None:
			# Use MPPI to combine actions
			R = plan_rews
			advs = (R - np.min(R)) / (np.max(R) - np.min(R) + 1e-6)
			S = np.exp(advs / self.params['mpc']['temp'])
			weighted_seq = S * actions.T
			planned_actions = np.sum(weighted_seq.T, axis=0)
			planned_actions /= np.sum(S) + 1e-6

			# We could also do CEM, etc. here
		else:
			# As temp --> 0, MPPI becomes random shooting
			ind = np.argmax(plan_rews)
			planned_actions = actions[ind]

		# Turn planned_actions back into a list
		self.planned_actions = []
		for i in range(planned_actions.shape[0]):
			self.planned_actions.append(planned_actions[i])

		# Calculate information for logging
		ro_mean = np.mean(plan_rews)
		ro_std = np.std(plan_rews)
		emp_mean = np.mean(plan_rews_emp)
		emp_std = np.std(plan_rews_emp)

		# Calculate information for MPC's generated plan
		env_state = self.env.sim.get_state() if self.mujoco else None
		fin_states, _, fin_rews, fin_rew, emp_rew = \
			traj.eval_traj(
				copy.deepcopy(self.env),
				env_state, self.prev_obs,
				mujoco=self.mujoco, perturb=self.perturb,
				H=len(self.planned_actions),
				gamma=self.gamma, act_mode='fixed',
				pt=(np.array(self.planned_actions), 0),
				terminal=terminal, tvel=self.tvel
			)

		# Store information in history
		self.hist['plan'][self.time].append(
			[ro_mean, ro_std, fin_rew, emp_mean, emp_std]
		)

		self.cache = (states, plan_rews, fin_states, fin_rews)

		return states, plan_rews

	def print_logs(self):
		"""
		MPC-specific logging information.
		"""
		bi, ei = super(MPCAgent, self).print_logs()

		self.print('MPC metrics', mode='head')
			
		# Print out planning levels
		plan_iters = [len(self.hist['plan'][i]) for i in range(bi, ei)]
		self.print('planning iters avg', np.mean(plan_iters))
		self.print('time horizon avg',
			np.mean(self.hist['H_des'][bi:ei]))

		# Print out initial planning information
		self.print('initial plan value',
			self.hist['init_plan'][self.time-1][0])
		self.print('initial plan emp rew',
			self.init_cache[2])

		# Print out information from prior usage
		if self.has_pol:
			num_prior = 0
			for t in range(bi, ei):
				if self.hist['prior_des'][t] == 'pol':
					num_prior += 1
			self.print('prior decision pct', num_prior / (ei-bi))

		# Print out information from optimization procedure
		if len(self.cache) >= 1:
			plan_rews = self.cache[1]
			if len(plan_rews) > 0:
				self.print('random plans max', np.max(plan_rews))
				self.print('random plans avg', np.mean(plan_rews))
				self.print('random plans std', np.std(plan_rews))

		if len(self.hist['plan'][self.time-1]) > 0:
			self.print('random plans emp rew avg',
				self.hist['plan'][self.time-1][-1][3])
			self.print('mpc plan rew',
				self.hist['plan'][self.time-1][-1][2])

		return bi, ei

	def advance_plan(self):
		"""
		Advance the old trajectory by a timestep to update it for the next
		timestep.
		"""
		action = self.planned_actions[0]
		self.planned_actions = self.planned_actions[1:]
		self.ghost_plan = self.ghost_plan[1:]
		self.extend_plan(self.H)
		return action

	def extend_plan(self, length=None):
		"""
		Extend the plan to meet a new length, and also keep a log of previous
		plans before extension, to reuse past computation.
		"""
		if length is None:
			length = self.H

		new_ghost = []
		for i in range(length):
			if i < len(self.planned_actions):
				# Update new ghost plan
				new_ghost.append(self.planned_actions[i])
			elif i < len(self.ghost_plan):
				# Use old ghost plan for new plan
				self.planned_actions.append(self.ghost_plan[i])
				new_ghost.append(self.ghost_plan[i])
			elif len(self.planned_actions) > 0:
				# No ghost plan available, just repeat action
				self.planned_actions.append(self.planned_actions[-1])
			else:
				# No past plan available, just use zero
				self.planned_actions.append(np.zeros(self.M))

		# Update the ghost plan if it has more information
		if len(new_ghost) >= len(self.ghost_plan):
			self.ghost_plan = new_ghost

		# Truncate the plan if it is too long
		self.planned_actions = self.planned_actions[:length]

	def do_updates(self):
		super(MPCAgent, self).do_updates()

	def use_paths(self, paths):
		return

	def finished_plan_update(self, iter_ind):
		return
