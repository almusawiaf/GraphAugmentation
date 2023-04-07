# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:55:13 2023

@author: almusawiaf
"""

import multiprocessing

# Define a function to compute the sum of a list of numbers
def sum_list(numbers):
    return sum(numbers)

# Define a list of numbers to sum
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Create a multiprocessing pool with 4 worker processes
with multiprocessing.Pool(processes=4) as pool:
    # Divide the list of numbers into chunks and apply the sum_list function to each chunk in parallel
    results = pool.map(sum_list, [numbers[i:i+3] for i in range(0, len(numbers), 3)])

# Compute the total sum of the results
total_sum = sum(results)

# Print the total sum
print(total_sum)

