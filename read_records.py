#!/usr/bin/env python3

"""
Script Name: read_records.py
Author: Stefan Herdy
Date: 13.03.2024
Description: 
Script to perform statistical analysis to compare the performance of the classification model with real and with generated data.
This script reads the records of the classification model and performs a t-test.
This enables conclusions to be drawn regarding how well the GAN recognizes and generates class-specific features,
as good recognition and generation of class-specific features should increase the accuracy of classification of the generated data compared to the real data

Usage: 
- First, run the classify.py script to train the model and generate the records.
- Make sure that you performed both classification trainings with real and generated data.
- Make sure that you have the records of the classification model in the records folder.
- Make sure that you have the same records path and the same number of tests as in the classify.py script.
"""

import json
import os
import numpy as np
from scipy import stats
from classify import tests
import argparse

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def get_closest_epoch_accuracy(json_data, epoch_number):
    closest_epoch = min(json_data.keys(), key=lambda x: abs(int(x) - epoch_number))
    
    if closest_epoch != str(epoch_number):
        print(f"Epoch {epoch_number} not found. Using closest available epoch {closest_epoch}.")
    
    return closest_epoch

def main(args):
    eval_results = {}
    for test in tests:
        record = {str(k): np.zeros(args.num_tests) for k, _ in read_json(os.path.join(args.records_path, os.listdir(args.records_path)[0])).items()}
        for i in range(args.num_tests):
            record_i = read_json(os.path.join(args.records_path, f'validation_accuracy_{test}_{i+1}.json'))
            for k in record_i.keys():
                record[k][i] += record_i[k]
        closest_epoch = get_closest_epoch_accuracy(record, args.eval_epoch)
        mean_record = {k: np.mean(v) for k, v in record.items()}
        std_record = {k: np.std(v) for k, v in record.items()}
        print(f'Standard deviation of test {test} at epoch {closest_epoch} is:')
        print(std_record[closest_epoch])
        print(f'Mean accuracy of test {test} at epoch {closest_epoch} is:')
        print(mean_record[closest_epoch])
        eval_results[test] = record[closest_epoch]


    test_norm = eval_results['norm']
    test_generated = eval_results['generated']

    t_statistic, p_value = stats.ttest_ind(test_norm, test_generated)

    print(f"t-statistic: {t_statistic}")
    print(f"p-value: {p_value}")

    if p_value < args.alpha:
        print("The difference is statistically significant (reject null hypothesis)")
    else:
        print("The difference is not statistically significant (fail to reject null hypothesis)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Statistics")
    parser.add_argument("--eval_epoch", type=int, default=200, help="Epoch to evaluate")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--num_tests", type=int, default=15, help="Number of tests to run")
    parser.add_argument("--records_path", type=str, default='./records', help="Path to records folder")
    args = parser.parse_args()
    main(args)

