env_params = {
    'hopper': {
        'changing': {
            'problem': {
                'T': 20000, 'gamma': .99, 'act_noise': 0,
                'ep_len': None, 'do_resets': False,
                'print_freq': 10, 'save_freq': 4000,
                'render_env': False, 'freeze': False,
                'dir_name': None
            },
            'env': {
                'env': 'Hopper', 'is_mujoco': True,
                'vel_schedule': [1.5, 2.5, 1, 3], 'vel_every': 4000,
                'tvel': True, 'vel_in_obs': False
            }
        },
        'novel': {
            'problem': {
                'T': 20000, 'gamma': .99, 'act_noise': 0,
                'ep_len': None, 'do_resets': False,
                'print_freq': 10, 'save_freq': 4000,
                'render_env': False, 'freeze': False,
                'dir_name': None
            },
            'env': {
                'env': 'Hopper', 'is_mujoco': True,
                'vel_schedule': [1.5, 2.5, 1, 3], 'vel_every': 4000,
                'tvel': True, 'vel_in_obs': True
            }
        },
        'standard': {
            'problem': {
                'T': 10000, 'gamma': .99, 'act_noise': 0,
                'ep_len': None, 'do_resets': False,
                'print_freq': 10, 'save_freq': 2500,
                'render_env': False, 'freeze': False,
                'dir_name': None
            },
            'env': {
                'env': 'Hopper', 'is_mujoco': True,
                'vel_schedule': [], 'vel_every': 100000,
                'tvel': False, 'vel_in_obs': False
            }
        }
    },
    'ant': {
        'changing': {
            'problem': {
                'T': 20000, 'gamma': .99, 'act_noise': 0,
                'ep_len': None, 'do_resets': False,
                'print_freq': 10, 'save_freq': 5000,
                'render_env': False, 'freeze': False,
                'dir_name': None
            },
            'env': {
                'env': 'Ant', 'is_mujoco': True,
                'vel_schedule': [], 'vel_every': 100000,
                'tvel': False, 'vel_in_obs': False,

                'perturb_schedule': [
                    {'type': 'zero', 'theta': 0, 'zero_inds': []},
                    {'type': 'zero', 'theta': 0, 'zero_inds': [0]},
                    {'type': 'zero', 'theta': 0, 'zero_inds': [2]},
                    {'type': 'zero', 'theta': 0, 'zero_inds': [6]},
                    {'type': 'zero', 'theta': 0, 'zero_inds': [1]},
                    {'type': 'zero', 'theta': 0, 'zero_inds': [7]},
                    {'type': 'zero', 'theta': 0, 'zero_inds': [2]},
                    {'type': 'zero', 'theta': 0, 'zero_inds': [3]}
                ], 'perturb_change_every': 2500
            }
        },
        'standard': {
            'problem': {
                'T': 10000, 'gamma': .99, 'act_noise': 0,
                'ep_len': None, 'do_resets': False,
                'print_freq': 10, 'save_freq': 2500,
                'render_env': False, 'freeze': False,
                'dir_name': None
            },
            'env': {
                'env': 'Ant', 'is_mujoco': True,
                'vel_schedule': [], 'vel_every': 100000,
                'tvel': False, 'vel_in_obs': False
            }
        }
    },
    'maze-d': {
        'changing': {
            'problem': {
                'T': 25000, 'gamma': .99, 'act_noise': 0,
                'ep_len': None, 'do_resets': False,
                'print_freq': 10, 'save_freq': 5000,
                'render_env': False, 'freeze': False,
                'dir_name': None
            },
            'env': {
                'env': 'Particle', 'is_mujoco': False,
                'pmaze_schedule': ['1', '16', '63', '100', '69', '30', '47', '52', '3', '8', '65', '88', '87', '78', '53', '10', '85', '28', '61', '12', '89', '86', '97', '18', '43', '98', '21', '84', '19', '6', '55', '92', '45', '68', '41', '14', '11', '74', '17', '64', '77', '62', '37', '32', '31', '48', '73', '96', '83', '2', '93', '72', '5', '80', '67', '70', '81', '58', '59', '40', '75', '34', '95', '54', '13', '50', '51', '20', '29', '26', '7', '90', '35', '94', '23', '60', '49', '4', '79', '46', '27', '38', '25', '22', '91', '42', '9', '82', '71', '24', '99', '44', '57', '56', '39', '66', '33', '76', '15', '36'],
                'pmaze_change_every': 500,
                'dense': True, 'pmaze_do_resets': False,

                'vel_schedule': [], 'vel_every': 100000,
                'tvel': False, 'vel_in_obs': False
            }
        },
        'novel': {
            'problem': {
                'T': 20000, 'gamma': .99, 'act_noise': 0,
                'ep_len': None, 'do_resets': False,
                'print_freq': 10, 'save_freq': 5000,
                'render_env': False, 'freeze': False,
                'dir_name': None
            },
            'env': {
                'env': 'Particle', 'is_mujoco': False,
                'pmaze_schedule': ['novel1', 'novel2'] * 5 \
                                + ['novel3', 'novel4'] * 5,
                'pmaze_change_every': 500,
                'dense': True, 'pmaze_do_resets': False,

                'vel_schedule': [], 'vel_every': 100000,
                'tvel': False, 'vel_in_obs': False
            }
        }
    },
    'maze-s': {
        'changing': {
            'problem': {
                'T': 25000, 'gamma': .99, 'act_noise': 0,
                'ep_len': None, 'do_resets': False,
                'print_freq': 10, 'save_freq': 5000,
                'render_env': False, 'freeze': False,
                'dir_name': None
            },
            'env': {
                'env': 'Particle', 'is_mujoco': False,
                'pmaze_schedule': ['1', '16', '63', '100', '69', '30', '47', '52', '3', '8', '65', '88', '87', '78', '53', '10', '85', '28', '61', '12', '89', '86', '97', '18', '43', '98', '21', '84', '19', '6', '55', '92', '45', '68', '41', '14', '11', '74', '17', '64', '77', '62', '37', '32', '31', '48', '73', '96', '83', '2', '93', '72', '5', '80', '67', '70', '81', '58', '59', '40', '75', '34', '95', '54', '13', '50', '51', '20', '29', '26', '7', '90', '35', '94', '23', '60', '49', '4', '79', '46', '27', '38', '25', '22', '91', '42', '9', '82', '71', '24', '99', '44', '57', '56', '39', '66', '33', '76', '15', '36'],
                'pmaze_change_every': 500,
                'dense': False, 'pmaze_do_resets': False,

                'vel_schedule': [], 'vel_every': 100000,
                'tvel': False, 'vel_in_obs': False
            }
        },
        'novel': {
            'problem': {
                'T': 20000, 'gamma': .99, 'act_noise': 0,
                'ep_len': None, 'do_resets': False,
                'print_freq': 10, 'save_freq': 5000,
                'render_env': False, 'freeze': False,
                'dir_name': None
            },
            'env': {
                'env': 'Particle', 'is_mujoco': False,
                'pmaze_schedule': ['novel1', 'novel2'] * 5 \
                                + ['novel3', 'novel4'] * 5,
                'pmaze_change_every': 500,
                'dense': False, 'pmaze_do_resets': False,

                'vel_schedule': [], 'vel_every': 100000,
                'tvel': False, 'vel_in_obs': False
            }
        }
    }
}
