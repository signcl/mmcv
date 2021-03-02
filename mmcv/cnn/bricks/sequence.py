from torch import nn

from .registry import SEQUENCE_LAYERS

@SEQUENCE_LAYERS.register_module()
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

def build_seq_layer(cfg, *args, **kwargs):
    """Build sequential layer.

    Args:
        cfg (None or dict): The lstm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding lstm layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding lstm layer.

    Returns:
        nn.Module: Created lstm layer.
    """
    if cfg is None:
        cfg_ = dict(type='BiLSTM')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in SEQUENCE_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        seq_layer = SEQUENCE_LAYERS.get(layer_type)

    layer = seq_layer(*args, **kwargs, **cfg_)

    return layer