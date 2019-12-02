import numpy as np
import torch
import copy
import multiprocessing as mp

def eval_traj(
    start_env, start_state, prev_obs, 
    mujoco=True, perturb=None,
    H=64, gamma=.99,
    act_mode='fixed', pt=(),
    terminal=None,
    tvel=None):
    """
    Evaluates a trajectory. Supports multiple modes of action generation:
    - 'fixed': fixed actions (e.g. MPC)
    - 'deter': deterministic policy (e.g. TD3)
    - 'gauss': Gaussian policy (e.g. VPG, PPO)
    Can also evaluate trajectories with terminal value function.
    tvel is specific to environments with a target velocity that must be set.
    """

    # Deepcopy of start_env should be done already
    N, env = prev_obs.shape[0], start_env

    # Initialize environment
    prev_state = prev_obs
    if mujoco:
        env.sim.set_state(start_state)
    else:
        env = copy.deepcopy(start_env)
    if tvel is not None:
        env.set_target_vel(tvel)

    # Trajectory logging/evaluation metrics
    cum_rew, fin_std, dis = 0, 0, 1
    states, rews = np.zeros((H, N)), np.zeros(H)
    acts = []
    
    # Run trajectory
    for t in range(H):
        if act_mode == 'fixed':
            # pt = (actions, filter_coefs)
            act = pt[0][t]
        elif act_mode == 'deter':
            # pt = (policy, pol_stds)
            ps = np.array(prev_state)
            act = pt[0].select_action(ps)
            if pt[1] is not None:
                act += np.random.normal(0, pt[1], size=act.shape)
        elif act_mode == 'gauss':
            # pt = (sampling function, policy, pol_stds)
            act = pt[0](pt[1], pt[2], prev_state)
        else:
            print('WARNING: act_mode not recognized')
            return

        # Perturb action and step with it
        perturbed_act = act
        if perturb is not None:
            perturbed_act = perturb.perturb(act)
        perturbed_act = np.clip(perturbed_act, -1, 1)
        states[t], rews[t], _, _ = env.step(perturbed_act)

        # Logging/evaluation
        cum_rew += dis * rews[t]
        dis *= gamma
        prev_state = states[t]
        acts.append(act)

    # Evaluate trajectory with terminal value function
    emp_rew = cum_rew
    if terminal is not None:
        terminal.eval()
        ps = torch.tensor(prev_state, dtype=terminal.dtype)
        term_val = terminal.forward(ps)
        if len(term_val.shape) > 0:
            term_val = term_val[0]
        cum_rew += dis * term_val.detach().cpu().numpy()

    return [states, np.array(acts), rews, cum_rew, emp_rew]

def generate_trajectories(
    num_rollouts,
    start_env, start_state, prev_obs,
    mujoco=True, perturb=None,
    H=64, gamma=.99,
    act_mode='fixed', pt=(),
    terminal=None,
    tvel=None,
    num_cpu=1):
    """
    Generates trajectories with multiprocessing. Should be used for action
    generation for MPC, policy gradient, etc.
    """
    rollouts_per_cpu = max(num_rollouts // num_cpu, 1)

    args_list = [
        rollouts_per_cpu,
        copy.deepcopy(start_env),
        copy.deepcopy(start_state),
        prev_obs,
        mujoco, perturb,
        H, gamma,
        act_mode, pt,
        terminal, tvel
    ]

    results = _try_multiprocess(
        args_list, num_cpu, 
        generate_paths_acts_star
    )

    # Combine all paths into one list
    paths = []
    for result in results:
        for path in result:
            paths.append(path)

    return paths

def do_rollouts_acts(
    num_rollouts,
    start_env, start_state, prev_obs,
    mujoco=True, perturb=None,
    H=64, gamma=.99,
    act_mode='fixed', pt=(),
    terminal=None,
    tvel=None, rseed=None):
    """
    Runs num_rollouts rollouts using eval_traj.
    """
    paths = []
    if act_mode == 'fixed':
        # pt = (actions, filter_coefs)
        plan = pt[0]
        lam, beta_0, beta_1, beta_2 = pt[1]

    if rseed is not None:
        np.random.seed(rseed)

    for i in range(num_rollouts):
        if act_mode == 'fixed':
            # Generate trajectories for MPC

            sigma = np.random.exponential(1/lam)

            eps = np.random.normal(
                loc=0, scale=1, size=pt[0].shape)
            eps *= sigma

            # Smooth noise temporally (O-U noise)
            for i in range(2, eps.shape[0]):
                eps[i] = beta_0*eps[i] + beta_1*eps[i-1] + beta_2*eps[i-2]

            # Apply noise to actions
            acts = np.clip(plan + eps, -1, 1)

            pti = (acts,)
        else:
            pti = pt

        paths.append(
            eval_traj(
                start_env, start_state, prev_obs, 
                mujoco=mujoco, perturb=perturb,
                H=H, gamma=gamma,
                act_mode=act_mode, pt=pti,
                terminal=terminal,
                tvel=tvel
        ))

    return paths

def generate_paths_acts_star(args_list):
    return do_rollouts_acts(*args_list)

def _try_multiprocess(args_list, num_cpu, f, max_timeouts=1):
    """
    Multiprocessing wrapper function.
    """
    if max_timeouts == 0:
        return None

    if num_cpu == 1:
        return [f(args_list)]
    else:
        pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
        pruns = []
        for _ in range(num_cpu):
            rseed = np.random.randint(1000000)
            pruns.append(pool.apply_async(f, args=(args_list+[rseed],)))
        try:
            results = [p.get(timeout=36000) for p in pruns]
        except Exception as e:
            print(str(e))
            print('WARNING: error raised in multiprocess, trying again')
            
            pool.close()
            pool.terminate()
            pool.join()

            return _try_multiprocess(args_list, num_cpu, f, max_timeouts-1)

        pool.close()
        pool.terminate()
        pool.join()

    return results
