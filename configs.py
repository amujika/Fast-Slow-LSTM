
def get_config(model):
    if model == 'enwik8':
        return enwik_config()
    if model == 'ptb':
        return ptb_config()
    else:
        raise ValueError("Invalid model: %s", model)

class enwik_config(object):
    """Enwik8 config."""
    init_scale = 0.01
    learning_rate = 0.001
    max_grad_norm = 1.0
    num_layers = 2
    num_steps = 100
    cell_size = 1200
    hyper_size = 1500
    embed_size = 256
    max_epoch = 35
    max_max_epoch = max_epoch
    keep_prob = 0.75
    zoneout_h = 0.95
    zoneout_c = 0.7
    lr_decay = 0.1
    batch_size = 128
    vocab_size = 205
    fast_layers = 4
    dataset = 'enwik8'


class ptb_config(object):
    """PTB config."""
    init_scale = 0.01
    learning_rate = 0.002
    max_grad_norm = 1.0
    num_layers = 2
    num_steps = 150
    cell_size = 700
    hyper_size = 400
    embed_size = 128
    max_epoch = 200
    max_max_epoch = max_epoch
    keep_prob = 0.65
    zoneout_h = 0.9
    zoneout_c = 0.5
    lr_decay = 0.1
    batch_size = 128
    vocab_size = 50
    fast_layers = 2
    dataset = 'ptb'
