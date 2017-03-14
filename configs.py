
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
    cell_size = 900
    hyper_size = 1500
    embed_size = 128
    max_epoch = 30
    max_max_epoch = 30
    keep_prob = 0.9
    in_k_prob = 0.9
    out_k_prob = 0.9
    zoneout_h = 0.95
    zoneout_c = 0.75
    lr_decay = 0.1
    batch_size = 64
    vocab_size = 205
    dataset = 'enwik8'


class ptb_config(object):
    """PTB config."""
    init_scale = 0.01
    learning_rate = 0.002
    max_grad_norm = 1.0
    num_layers = 2
    num_steps = 100
    cell_size = 1000
    hyper_size = 400
    embed_size = 128
    max_epoch = 100
    max_max_epoch = 100
    keep_prob = 0.5
    in_k_prob = 0.5
    out_k_prob = 0.5
    zoneout_h = 0.85
    zoneout_c = 0.5
    lr_decay = 0.1
    batch_size = 128
    vocab_size = 50
    dataset = 'ptb'
