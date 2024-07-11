"""
Sorting Movies By Duration Time
You have been asked to sort movies according to their duration in descending order.

Your output should contain all columns sorted by the movie duration in the given dataset.

movie_catalogue
show_id	title	                release_year	rating	duration
s1	    Dick Johnson Is Dead	2020	        PG-13	  90 min
s95	    Show Dogs	            2018	        PG	    90 min
s108	  A Champion Heart	    2018	        G	      90 min
"""

# Import your libraries
import pandas as pd

# Start writing code
movie_catalogue.head()

# convert duration from string to integer for sorting
movie_catalogue['duration'] = movie_catalogue['duration'].str.replace('min', '').astype(int)

# sort the Duration by duration in descending order
sorted_movie_catalogue = movie_catalogue.sort_values(by='duration', ascending=False)

# convert duration back to string
sorted_movie_catalogue['duration'] = sorted_movie_catalogue['duration'].astype(str) + 'min'

# print sorted dataset
sorted_movie_catalogue
