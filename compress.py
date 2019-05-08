# Run this script from the terminal as follows:
# python compress.py input_dir output_dir
# replacing input_dir by the directory containing the *.tif images
# to compress, and output_dir by the destination folder

# Requires GDAL installed in the machine!



import os
import subprocess
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

for file in os.listdir(input_dir):
    if file.endswith(".tif"):
        input_file = os.path.join(input_dir,file)
        output_file = os.path.join(output_dir,file)
        command = "gdal_translate --config GDAL_PAM_ENABLED NO -co COMPRESS=CCITTFAX4 -co NBITS=1 "+input_file+" "+output_file
        print(command)
        subprocess.call(command,shell=True)

print ("Done!")
