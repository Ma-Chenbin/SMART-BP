## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class MIMIC():
    def __init__(self):
        super(MIMIC, self).__init__()
        self.train_params = {
            'num_epochs': 200,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            "CNN": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "TCN": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "GRU": {
                "lambda": 1.9545162611489,
                "mu": 1.98498416516161,
                "phi": 1.,
                "alpha": 1.,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "SASA": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "domain_loss_wt": 7.3937939938562,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "ResNet": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "domain_loss_wt": 7.3937939938562,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "DDC": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "learning_rate": 0.001,
                "mmd_wt": 3.7991920933520342,
                "src_cls_loss_wt": 6.286301875125623,
                "domain_loss_wt": 6.36,
                "weight_decay": 0.0001
            },
            "CoDATS": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "domain_loss_wt": 3.2750474868706925,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 6.335109786953256,
                "weight_decay": 0.0001
            },
            "Inception": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "domain_loss_wt": 3.2750474868706925,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 6.335109786953256,
                "weight_decay": 0.0001
            },
            "Xception": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "domain_loss_wt": 7.3937939938562,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "DANN": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "domain_loss_wt": 2.943729820531079,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 5.1390077646202,
                "weight_decay": 0.0001
            },
            "DIRT": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "cond_ent_wt": 1.20721518968644,
                "domain_loss_wt": 1.9012145515129044,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 9.67861021290254,
                "vat_loss_wt": 7.7102843136045855,
                "weight_decay": 0.0001
            },
            "mWDN": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "domain_loss_wt": 7.3937939938562,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "DSAN": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "learning_rate": 0.001,
                "mmd_wt": 2.0872340713147786,
                "src_cls_loss_wt": 1.8744909939900247,
                "domain_loss_wt": 1.59,
                "weight_decay": 0.0001
            },
            "GRU_RCN": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "domain_loss_wt": 7.3937939938562,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "MMDA": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "cond_ent_wt": 1.383002023133561,
                "coral_wt": 8.36810764913737,
                "learning_rate": 0.001,
                "mmd_wt": 3.964042918489996,
                "src_cls_loss_wt": 6.794522068759213,
                "weight_decay": 0.0001
            },
            "Transformer": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "domain_loss_wt": 7.3937939938562,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "Deep_Coral": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "coral_wt": 4.23035475456397,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.1013209750429822,
                "weight_decay": 0.0001
            },
            "KNN": {
                "n_neighbors": 3,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "SVR": {
                "kernel": 'rbf',
                "C": 1.,
                "epsilon": 0.1,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "RF": {
                "n_estimators": 100,
                "max_depth": 9.,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "XGBoost": {
                "n_estimators": 100,
                "max_depth": 9.,
                "learning_rate": 0.005,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "Stacking": {
                "learning_rate": 0.001,
                "src_cls_loss_wt": 4.185814373345016,
                "weight_decay": 0.0001
            },
            "MLP": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "cond_ent_wt": 1.383002023133561,
                "coral_wt": 8.36810764913737,
                "learning_rate": 0.001,
                "mmd_wt": 3.964042918489996,
                "src_cls_loss_wt": 6.794522068759213,
                "weight_decay": 0.0001
            },
            "CDAN": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "cond_ent_wt": 1.2920143348777362,
                "domain_loss_wt": 9.545761950873414,
                "learning_rate": 0.001,
                "src_cls_loss_wt": 9.430292987535724,
                "weight_decay": 0.0001
            },
            "AdvSKM": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "domain_loss_wt": 1.338788378230754,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 2.468525942065072,
                "weight_decay": 0.0001
            },
            "HoMM": {
                "lambda": 1.445781546778,
                "mu": 1.23578177845,
                "phi": 1.,
                "alpha": 1.,
                "hommd_wt": 2.8305712579412683,
                "learning_rate": 0.0005,
                "src_cls_loss_wt": 0.1282520874653523,
                "domain_loss_wt": 9.13,
                "weight_decay": 0.0001
            },
            'CoTMix': {'learning_rate': 0.001, 'mix_ratio': 0.9, 'temporal_shift': 14,
                       'src_cls_weight': 0.78, 'src_supCon_weight': 0.1, 'trg_cont_weight': 0.1,
                       'trg_entropy_weight': 0.05, "lambda": 1.445781546778, "mu": 1.23578177845,
                       "phi": 1., "alpha": 1., },
            'MCD': {'learning_rate': 1e-2, 'src_cls_loss_wt': 9.74, 'domain_loss_wt': 5.43,
                    "lambda": 1.445781546778, "mu": 1.23578177845, "phi": 1., "alpha": 1., },

        }

