import os
import torch
import collections
import pandas as pd
import numpy as np
from algorithms.algorithms import get_algorithm_class
from utils import fix_randomness, starting_logs, AverageMeter

class AbstractTrainer:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset  # Dataset object with methods to get data loaders
        self.da_method = args.da_method  # Domain adaptation method name
        self.exp_log_dir = args.exp_log_dir  # Experiment log directory
        self.num_runs = args.num_runs  # Number of runs for each scenario
        self.dataset_configs = args.dataset_configs  # Dataset configurations
        self.home_path = args.home_path  # Home directory path
        self.results_columns = ["sbp", "dbp", "map", "run", 'me', 'sde', 'mae']  # Result metrics columns
        self.risks_columns = ["risk1", "risk2", "risk3", "run"]  # Risk metrics columns

    def load_data(self, src_id, trg_id):
        """
        Load data from source and target domains.

        Args:
            src_id (str): Source domain identifier.
            trg_id (str): Target domain identifier.
        """
        # Load source domain training data
        self.src_train_dl = self.dataset.get_dataloader(
            domain_id=src_id,
            split='train',
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )

        # Load target domain training data
        self.trg_train_dl = self.dataset.get_dataloader(
            domain_id=trg_id,
            split='train',
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )

        # Load target domain testing data
        self.trg_test_dl = self.dataset.get_dataloader(
            domain_id=trg_id,
            split='test',
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )

    def initialize_algorithm(self):
        """
        Initialize the domain adaptation algorithm based on the specified method.
        """
        AlgorithmClass = get_algorithm_class(self.da_method)
        self.algorithm = AlgorithmClass(self.args)

    def save_checkpoint(self, home_path, scenario_log_dir, last_model, best_model):
        """
        Save model checkpoints.

        Args:
            home_path (str): Home directory path.
            scenario_log_dir (str): Directory for saving logs and checkpoints.
            last_model (nn.Module): Model from the last training epoch.
            best_model (nn.Module): Model with the best validation performance.
        """
        last_model_path = os.path.join(scenario_log_dir, 'last_model.pth')
        best_model_path = os.path.join(scenario_log_dir, 'best_model.pth')
        torch.save(last_model.state_dict(), last_model_path)
        torch.save(best_model.state_dict(), best_model_path)

    def load_checkpoint(self, scenario_log_dir):
        """
        Load model checkpoints.

        Args:
            scenario_log_dir (str): Directory containing the checkpoints.

        Returns:
            tuple: State dictionaries of the last and best models.
        """
        last_model_path = os.path.join(scenario_log_dir, 'last_model.pth')
        best_model_path = os.path.join(scenario_log_dir, 'best_model.pth')

        # Load the state dictionaries
        last_chk = torch.load(last_model_path, map_location=self.args.device)
        best_chk = torch.load(best_model_path, map_location=self.args.device)
        return last_chk, best_chk

    def evaluate(self, dataloader):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (DataLoader): DataLoader for the evaluation dataset.
        """
        self.algorithm.network.eval()  # Set the model to evaluation mode
        self.all_preds = []
        self.all_targets = []

        with torch.no_grad():
            for data in dataloader:
                inputs, targets = data
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)

                # Forward pass
                outputs = self.algorithm.network(inputs)
                preds = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

                # Store predictions and targets
                self.all_preds.append(preds)
                self.all_targets.append(targets)

        # Concatenate all predictions and targets
        self.all_preds = np.concatenate(self.all_preds, axis=0)
        self.all_targets = np.concatenate(self.all_targets, axis=0)

    def calculate_metrics(self):
        """
        Calculate evaluation metrics based on the predictions and targets.

        Returns:
            dict: Dictionary containing the calculated metrics.
        """
        # Compute errors
        errors = self.all_preds - self.all_targets

        # Mean Error (ME)
        me = np.mean(errors)

        # Standard Deviation of Error (SDE)
        sde = np.std(errors)

        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(errors))

        # Add other metrics if needed, e.g., Root Mean Square Error (RMSE)
        # rmse = np.sqrt(np.mean(errors ** 2))

        metrics = {
            'me': me,
            'sde': sde,
            'mae': mae,
            # 'rmse': rmse,
        }
        return metrics

    def calculate_risks(self):
        """
        Calculate risks based on the evaluation metrics or other criteria.

        Returns:
            dict: Dictionary containing the calculated risks.
        """
        # Example risk calculations based on thresholds
        errors = self.all_preds - self.all_targets
        abs_errors = np.abs(errors)

        # Risk 1: Percentage of samples with absolute error > 5 mmHg
        risk1 = np.mean(abs_errors > 5) * 100  # Convert to percentage

        # Risk 2: Percentage of samples with absolute error > 10 mmHg
        risk2 = np.mean(abs_errors > 10) * 100

        # Risk 3: Custom risk metric (e.g., combined risk)
        risk3 = (risk1 + risk2) / 2

        risks = {
            'risk1': risk1,
            'risk2': risk2,
            'risk3': risk3,
        }
        return risks

    def append_results_to_tables(self, table, scenario, run_id, metrics):
        """
        Append results to the table.

        Args:
            table (DataFrame): Pandas DataFrame to append results to.
            scenario (str): Scenario identifier (e.g., 'A_to_B').
            run_id (int): Identifier for the current run.
            metrics (dict): Dictionary containing metrics to append.

        Returns:
            DataFrame: Updated DataFrame with the new results appended.
        """
        result_row = {'scenario': scenario, 'run': run_id}
        result_row.update(metrics)
        table = table.append(result_row, ignore_index=True)
        return table

    def add_mean_std_table(self, table, columns):
        """
        Add mean and standard deviation rows to the table.

        Args:
            table (DataFrame): Pandas DataFrame containing the results.
            columns (list): List of columns to calculate mean and std for.

        Returns:
            DataFrame: Updated DataFrame with mean and std rows appended.
        """
        # Calculate mean and std for specified columns
        mean_values = table[columns].mean()
        std_values = table[columns].std()

        # Create mean and std rows
        mean_row = {'scenario': 'mean', 'run': '-'}
        mean_row.update(mean_values.to_dict())

        std_row = {'scenario': 'std', 'run': '-'}
        std_row.update(std_values.to_dict())

        # Append to the table
        table = table.append(mean_row, ignore_index=True)
        table = table.append(std_row, ignore_index=True)
        return table

    def save_tables_to_file(self, table, filename):
        """
        Save the table to a CSV file.

        Args:
            table (DataFrame): Pandas DataFrame to save.
            filename (str): Name of the file (without extension).
        """
        filepath = os.path.join(self.exp_log_dir, f"{filename}.csv")
        table.to_csv(filepath, index=False)
