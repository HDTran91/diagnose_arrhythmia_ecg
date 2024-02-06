import sys
import json

# Receive data from Node.js through command line arguments
data_from_nodejs = json.loads(sys.argv[1])

print(data_from_nodejs)

# Process the data (in this case, simply print it)
for index, filename in enumerate(data_from_nodejs):
    print(f"Image {index + 1} received - Filename: {filename}")
