import sys
import os

if len(sys.argv) < 3:
    print("Incorrect syntax")
    exit(1)

directory = sys.argv[1]
output_path = sys.argv[2]

# Get a list of all files in the directory and prepend the directory path
files = [directory + filename for filename in os.listdir(directory)]

# Open up the output file
# We want to write incrementally, rather than store all data
# in memory
output_file = open(output_path, "w+")
output_file.write("movieId,userId,rating,timestamp\n")

for filepath in files:
    lines = open(filepath, "r", encoding="latin-1").readlines()
    movie_id = lines[0].strip().replace(":", "")
    print("\rWriting " + str(movie_id), end="")
    for rating in lines[1:]:
        output_file.write(movie_id + "," + rating.strip() + "\n")

print("\n")
