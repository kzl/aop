"""
Default listing of parameters used for Adaptive Online Planning (AOP).

Agents should be initialized with params; some experiments may have slightly
different settings, but most non-environment parameters are held constant.
"""

base_params = {

# ==============================================
#  ADAPTIVE ONLINE PLANNING (AOP) METHODS
#   - AOP-TD3
#   - AOP-BC
#   * note that these borrow hyperparameters
#     from other sections (e.g. MPC, POLO)
# ==============================================

    'aop': {
        'std_thres': 8, 'bellman_thres': 25,
        'ratio_thres': .05, 'init_thres': .01,
        'eps_plan': .2
    },
    'p-td3': {
        'update_every': 4, 'buf_size': 100000,
        'grad_steps': 128, 'hidden_sizes': [400,300]
    },
    'p-bc': {
        'update_freq': 4,
        'grad_steps': 400, 'batch_size': 64,
        'buf_size': 10000, 'pb_pct': .8,
        'pol_optim': {
            'lr': 1e-3, 'reg': 0, 'ent_temp': 0
        }, 'pol_params': {
            'activation': 'tanh', 'dropout': 0,
            'hidden_sizes': [400,300]
        },
    },

# ==============================================
#  MODEL-BASED METHODS
#   - Online MPC
#   - POLO
# ==============================================

    'mpc': {
        'H': 80, 'num_rollouts': 40, 'num_iter': 8,
        'filter_coefs': (.1,.05,.8,0), 'temp': .01,
        'use_terminal': True, 'use_pol_anyway': False,
        'num_cpu': 1, 'print_iter_vals': False
    },
    'polo': {
        'update_freq': 4, 'H_backup': 64, 'buf_size': 1000,
        'grad_steps': 32, 'batch_size': 32,
        'ens_params': {
            'ens_size': 6, 'lr': 1e-3, 'reg': 0,
            'prior_beta': 1, 'kappa': 1e-2, 'rpf_noise': 0, 
            'model_params': {
                'activation': 'tanh', 'dropout': 0,
                'hidden_sizes': [64,64]
            }
        }
    },

# ==============================================
#  MODEL-FREE METHODS
#   - Online VPG
#   - Online PPO
#   - Online TD3
# ==============================================

    'pg': {
        'H': 128, 'update_every': 1,
        'num_iter': 1, 'num_rollouts': 32,
        'pol_steps': 80, 'val_steps': 80,
        'batch_size': 4096,
        'use_gae': True, 'lam': .95,
        'min_std': .2, 'max_std': .8,
        'grad_clip': 10,
        'run_deterministic': True,
        'num_cpu': 1, 'print_per_iters': 1,
        'pol_optim': {
            'lr': 3e-4, 'reg': 0, 'ent_temp': 0
        }, 'pol_params': {
            'activation': 'tanh', 'dropout': 0,
            'hidden_sizes': [64,64]
        }, 'val_optim': {
            'lr': 1e-3, 'reg': 0
        }, 'val_params': {
            'activation': 'tanh', 'dropout': 0,
            'hidden_sizes': [64,64]
        }
    },
    'ppo': {
        'eps': .2, 'target_kl': 10
    },
    'td3': {
        'H': 256, 'update_every': 1,
        'num_iter': 1, 'num_rollouts': 1,
        'grad_steps': 256, 'act_std': .2,
        'hidden_sizes': [400,300], 'buf_size': 100000,
        'num_cpu': 1, 'print_per_iters': 2,
    }
}
