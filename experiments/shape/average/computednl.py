"""
Script to compute DNL for benchmark test cases.

This script loads a benchmark dataset, computes the DNL for each test case,
and saves the updated benchmark data with DNL values.
"""

import sys
sys.path.insert(1, '../../../eispy2d/library/')
import benchmark as bmk

# Load the benchmark dataset from the specified file and path
benchmark = bmk.Benchmark(import_filename="average.bmk",
                          import_filepath="../../../data/shape/average/")

# Compute DNL for each test case in the benchmark
for n in range(30):
    benchmark.testset.test[n].compute_dnl()

# Save the updated benchmark data with computed DNL values
# save_testset=True ensures that the test set data is also saved
benchmark.save(file_path="../../../data/shape/average/",
               save_testset=True)
