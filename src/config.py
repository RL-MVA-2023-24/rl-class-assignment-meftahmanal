import torch
config = {
    'nb_actions': None, 
    'learning_rate': 0.001,
    'gamma': 0.95,
    'buffer_size': 200000,
    'epsilon_min': 0.01,
    'epsilon_max': 1,
    'epsilon_decay_period': 20000,
    'epsilon_delay_decay': 20,
    'batch_size': 2000,
    'gradient_steps': 5,
    'update_target_strategy': 'ema',
    'update_target_freq': 50,
    'update_target_tau': 0.0005,
    'criterion': torch.nn.SmoothL1Loss(),
    'monitoring_nb_trials': 0,
}
