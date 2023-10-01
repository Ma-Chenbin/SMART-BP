def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class BloodPressureRegression():
    def __init__(self):
        super(BloodPressureRegression, self).__init__()

        self.class_names = ['sbp', 'dbp', 'map']
        self.scenarios = [("sbp", "dbp"), ]
        self.num_classes = 1

        # Data processing parameters
        self.sequence_len = 1125
        self.normalize = True
        self.shuffle = True
        self.drop_last = True

        # Model Configuration Parameters
        self.input_channels = 1
        self.kernel_size = 3
        self.stride = 1
        self.dropout = 0.5
        self.hidden_dim = 64
        self.num_classes = 2  # SBP and DBP

        # CNN features
        self.mid_channels = 32
        self.final_out_channels = 64
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # RNN features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128

class MIMIC():
    def __init__(self):
        super(MIMIC, self).__init__()

        self.class_names = ['sbp', 'dbp', 'map']
        self.scenarios = [("sbp", "dbp"), ]
        self.num_classes = 1

        # Data processing parameters
        self.sequence_len = 1125
        self.normalize = True
        self.shuffle = True
        self.drop_last = True

        # Model Configuration Parameters
        self.input_channels = 1
        self.kernel_size = 3
        self.stride = 1
        self.dropout = 0.5
        self.hidden_dim = 64
        self.num_classes = 2  # SBP and DBP

        # CNN features
        self.mid_channels = 32
        self.final_out_channels = 64
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # RNN features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128

class Mindray():
    def __init__(self):
        super(Mindray, self).__init__()

        self.class_names = ['sbp', 'dbp', 'map']
        self.scenarios = [("sbp", "dbp"), ]
        self.num_classes = 1

        # Data processing parameters
        self.sequence_len = 1125
        self.normalize = True
        self.shuffle = True
        self.drop_last = True

        # Model Configuration Parameters
        self.input_channels = 1
        self.kernel_size = 3
        self.stride = 1
        self.dropout = 0.5
        self.hidden_dim = 64
        self.num_classes = 2  # SBP and DBP

        # CNN features
        self.mid_channels = 32
        self.final_out_channels = 64
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # RNN features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128

class CAS_BP():
    def __init__(self):
        super(CAS_BP, self).__init__()

        self.class_names = ['sbp', 'dbp', 'map']
        self.scenarios = [("sbp", "dbp"), ]
        self.num_classes = 1

        # Data processing parameters
        self.sequence_len = 1125
        self.normalize = True
        self.shuffle = True
        self.drop_last = True

        # Model Configuration Parameters
        self.input_channels = 1
        self.kernel_size = 3
        self.stride = 1
        self.dropout = 0.5
        self.hidden_dim = 64
        self.num_classes = 2  # SBP and DBP

        # CNN features
        self.mid_channels = 32
        self.final_out_channels = 64
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # RNN features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128