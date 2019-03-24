"""
Script to evaluate model predictions based on mean quadratic weighted kappa
Columns in csv file: 'target', 'output', 'set'
"""

from ml_metrics import quadratic_weighted_kappa, mean_quadratic_weighted_kappa
import pandas as pd


def calc_mqwp(output):
    """
    Calculate the mean quadratic_weighted_kappa across all the question sets
    :param outputs: dataframe containing target, output, question set
    :return: mean quadratic weighted kappa
    """
    groups = output.groupby('set')

    kappas = [quadratic_weighted_kappa(group[1]["output"], group[1]["target"])
              for group in groups]
    print('Kappa of each set: ', kappas)
    mean = mean_quadratic_weighted_kappa(kappas)
    return mean


if __name__ == '__main__':
    output = pd.read_csv('outputs.csv')

    qwp = quadratic_weighted_kappa(output['output'], output['target'])
    mqwp = calc_mqwp(output)

    print('Quadratic weighted average: ', qwp)
    print('Mean quadratic weighted average: ', mqwp)
