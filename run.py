import argparse
import copy

import params.default_params as default_params
import params.env_params as env_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', '-a', type=str, default='aop',
        choices=['aop', 'aop-bc', 'polo', 'td3', 'ppo', 'mpc-8', 'mpc-3'],
        help='Choice of algorithm to use for training')
    parser.add_argument('--env', '-e', type=str, default='hopper',
        choices=['hopper', 'ant', 'maze-d', 'maze-s'],
        help='Base environment for agent')
    parser.add_argument('--setting', '-s', type=str, default='changing',
        choices=['changing', 'novel', 'standard'],
        help='Specify which setting to test in')
    parser.add_argument('--output_dir', '-d', type=str,
        help='Directory in ex/ to output models to (for example, ex/my_exp_1)')
    parser.add_argument('--num_trials', '-n', type=int, default=1,
        help='Number of trials (seeds) to run for')
    parser.add_argument('--num_cpus', '-c', type=int, default=4,
        help='Number of CPUs to use for trajectory generation')
    parser.add_argument('--use_gpu', '-g', default=True,
        help='Whether or not to use GPU (currently only TD3 supports this)')
    parser.add_argument('--test_pol', '-t', default=True,
        help='Whether or not to test the policy in standard episode')

    args = parser.parse_args()

    if not is_valid_env(args.env, args.setting):
        print('Environment \"%s %s\" is not supported, terminating'
                % (args.setting, args.env))
        return

    # Basic information for experiments

    agent_class = get_agent_class(args.algo)

    output_dir = args.output_dir if args.output_dir else default_output_dir()

    # Setting parameter settings for experiments

    params = copy.deepcopy(default_params.base_params)
    params.update(env_params.env_params[args.env][args.setting])

    params['problem']['algo'] = args.algo
    params['problem']['output_dir'] = output_dir

    params['mpc']['num_cpu'] = args.num_cpus
    params['pg']['num_cpu'] = args.num_cpus

    params['problem']['test_pol'] = args.test_pol
    params['problem']['eval_len'] = 1000
    params['problem']['use_gpu'] = args.use_gpu

    # Setting environment-specific hyperparameter settings

    if args.env == 'maze-s':
        params['aop']['std_thres'] = 0
        params['aop']['bellman_thres'] = 0
    elif args.env == 'ant':
        params['aop']['ratio_thres'] = .01
        params['aop']['init_thres'] = -1

    if 'maze' in args.env:
        params['p-td3']['hs'] = [64,64]
        params['p-bc']['hs'] = [64,64]
        params['td3']['hs'] = [64,64]

    # Setting algorithm-specific hyperparameter settings

    if args.algo == 'polo' or args.algo == 'mpc-3':
        params['mpc']['num_iter'] = 3

    # Run experiments

    for i in range(args.num_trials):
        params['problem']['dir_name'] = '%s/trial_%d' % (output_dir, i)
        agent = agent_class(params)
        agent.run_lifetime()

def is_valid_env(env_name, setting):
    if env_name == 'hopper':
        return True
    elif env_name == 'ant' and setting in ['changing', 'standard']:
        return True
    elif 'maze' in env_name and setting in ['changing', 'novel']:
        return True
    else:
        return False

def default_output_dir():
    import datetime
    now = datetime.datetime.now()
    ctime = '%02d%02d_%02d%02d' % (now.month, now.day, now.hour, now.minute)
    return 'ex/' + ctime

def get_agent_class(algo):
    if algo == 'aop':
        from agents.AOPTD3Agent import AOPTD3Agent
        agent_class = AOPTD3Agent
    elif algo == 'aop-bc':
        from agents.AOPBCAgent import AOPBCAgent
        agent_class = AOPBCAgent
    elif algo == 'polo':
        from agents.POLOAgent import POLOAgent
        agent_class = POLOAgent
    elif algo == 'td3':
        from agents.TD3Agent import TD3Agent
        agent_class = TD3Agent
    elif algo == 'ppo':
        from agents.PPOAgent import PPOAgent
        agent_class = PPOAgent
    elif algo == 'mpc-8' or algo == 'mpc-3':
        from agents.MPCAgent import MPCAgent
        agent_class = MPCAgent
    return agent_class

if __name__ == '__main__':
    main()
