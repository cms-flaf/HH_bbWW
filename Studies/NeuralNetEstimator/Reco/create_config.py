import io 
import yaml

cfg = {
    'topology' : [256, 128, 64, 32, 16, 8],
    'learning_rate' : 0.001,
    'n_epochs' : 50,
    'batch_size' : 256,
    'verbosity' : 1,
    'valid_split' : 0.33,
    'name' : 'nn_estimator_v1'
}

with io.open('config.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(cfg, outfile, default_flow_style=False, allow_unicode=True)