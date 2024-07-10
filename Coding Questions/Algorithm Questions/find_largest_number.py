"""
Maximum Number in Array
Find the largest number in an array.

Constraints
The input variable arr is a list of integers.
The list arr can have any length.
The integers in the list arr can be positive, negative, or zero.
The integers in the list arr can be in any order.
The list arr can contain duplicate integers.
The list arr cannot be empty.

Test Case #1
Input: [3, -2, 7, 3, -9, 12]
Output: 12
Description: This test case includes a mix of positive, negative, and duplicate numbers to ensure it finds the highest positive number.
Test Case #2
Input: [-15, -22, -3, -7, -9, -12]
Output: -3
Description: This test case comprises only negative numbers, challenging the algorithm to identify the least negative number (which is the largest in this context).
Test Case #3
Input: [1000000, -1000000, 0, 2, -100, 500]
Output: 1000000
Description: This test case incorporates large positive and negative numbers, testing the algorithm""s ability to correctly handle numbers of varying magnitudes and find the largest positive integer.
"""

def find_largest_number(arr):
    """ 
    :type arr: List[int] 
    :rtype: int 
    """
    # Ensure input list is not empty
    if not arr:
        raise ValueError("The input list cannot be empty")
        
    # Initialize the largest number with the first element in the list
    largest_number = arr[0]
    
    # Iterate through the list starting from the second element
    for num in arr[1:]:
        if num > largest_number:
            largest_number = num
            
    return largest_number
    
    
