from trainers.train import Trainer

import argparse
parser = argparse.ArgumentParser()

if __name__ == "__main__":

    # ========  Experiments Phase ================
    parser.add_argument('--phase', default='train', type=str, help='train, test')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='experiments_logs', type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name', default='EXP1', type=str, help='experiment name')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'../datasets/MIMIC', type=str, help='Path containing dataset')
    parser.add_argument('--dataset', default='MIMIC', type=str, help='Dataset of choice: (MIMIC - Mindray - CAS_BP)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - ResNet - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=5, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')

    # arguments
    args = parser.parse_args()

    # create trainier object
    trainer = Trainer(args)

    # train and test
    if args.phase == 'train':
        trainer.fit()
    elif args.phase == 'test':
        trainer.test()



#TODO:
# 1- Change the naming of the functions ---> ( Done)
# 2- Change the algorithms following UPR-BP --> (Done)
# 3- Keep one trainer for both train and test --> (Done)
# 4- Create the new joint loader that consider the all possible batches --> Done
# 5- Implement Lower/Upper Bound Approach --> Done
# 6- Add the best hparams --> Done
# 7- Add pretrain based methods (ADDA, MCD, MDD)
