import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def vis_sbp(est_sbp, ref_sbp):
    # Compute the mean and difference between the estimated and reference SBP
    mean_sbp = np.mean([est_sbp, ref_sbp], axis=0)
    diff_sbp = est_sbp - ref_sbp

    # Compute the limits of agreement (LoA) as mean difference +/- 1.96 standard deviation of differences
    sd_diff = np.std(diff_sbp)
    # Compute the mean difference between the estimated and reference SBP
    mean_diff = np.mean(est_sbp - ref_sbp)
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    # Compute the Pearson correlation coefficient and root mean squared error between the estimated and reference SBP
    corr_coef, _ = pearsonr(est_sbp, ref_sbp)
    rmse = np.sqrt(mean_squared_error(est_sbp, ref_sbp))

    # Create the first plot (Bland-Altman plot)
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_sbp, diff_sbp, alpha=0.5)
    plt.axhline(y=np.mean(diff_sbp), color='black', linestyle='dashed')
    plt.axhline(y=loa_upper, color='red', linestyle='dashed')
    plt.axhline(y=loa_lower, color='red', linestyle='dashed')
    plt.xlabel('Mean of estimated and reference SBP (mmHg)')
    plt.ylabel('Difference between estimated and reference SBP (mmHg)')
    plt.title('Bland-Altman plot')

    # Create the second plot (correlation plot)
    plt.figure(figsize=(8, 6))
    plt.scatter(est_sbp, ref_sbp, alpha=0.5)
    plt.plot([np.min((est_sbp, ref_sbp)), np.max((est_sbp, ref_sbp))],
             [np.min((est_sbp, ref_sbp)), np.max((est_sbp, ref_sbp))],
             color='black', linestyle='dashed')
    plt.xlabel('Estimated SBP (mmHg)')
    plt.ylabel('Reference SBP (mmHg)')
    plt.title(f'Correlation plot (corr coef = {corr_coef:.2f}, RMSE = {rmse:.2f} mmHg)')



def vis_dbp(est_dbp, ref_dbp):
    # Compute the mean and difference between the estimated and reference DBP
    mean_dbp = np.mean([est_dbp, ref_dbp], axis=0)
    diff_dbp = est_dbp - ref_dbp

    # Compute the limits of agreement (LoA) as mean difference +/- 1.96 standard deviation of differences
    sd_diff = np.std(diff_dbp)
    # Compute the mean difference between the estimated and reference DBP
    mean_diff = np.mean(est_dbp - ref_dbp)
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    # Compute the Pearson correlation coefficient and root mean squared error between the estimated and reference DBP
    corr_coef, _ = pearsonr(est_dbp, ref_dbp)
    rmse = np.sqrt(mean_squared_error(est_dbp, ref_dbp))

    # Create the first plot (Bland-Altman plot)
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_dbp, diff_dbp, alpha=0.5)
    plt.axhline(y=np.mean(diff_dbp), color='black', linestyle='dashed')
    plt.axhline(y=loa_upper, color='red', linestyle='dashed')
    plt.axhline(y=loa_lower, color='red', linestyle='dashed')
    plt.xlabel('Mean of estimated and reference DBP (mmHg)')
    plt.ylabel('Difference between estimated and reference DBP (mmHg)')
    plt.title('Bland-Altman plot')

    # Create the second plot (correlation plot)
    plt.figure(figsize=(8, 6))
    plt.scatter(est_dbp, ref_dbp, alpha=0.5)
    plt.plot([np.min((est_dbp, ref_dbp)), np.max((est_dbp, ref_dbp))],
             [np.min((est_dbp, ref_dbp)), np.max((est_dbp, ref_dbp))],
             color='black', linestyle='dashed')
    plt.xlabel('Estimated DBP (mmHg)')
    plt.ylabel('Reference DBP (mmHg)')
    plt.title(f'Correlation plot (corr coef = {corr_coef:.2f}, RMSE = {rmse:.2f} mmHg)')