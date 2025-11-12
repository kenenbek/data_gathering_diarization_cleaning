import csv
import re

# Input and output file paths
input_file = 'diarization.txt'
output_file = 'diarization.csv'

# Regex pattern to match the diarization lines
pattern = r'(SPEAKER_\d+) speaks between t=(\d+\.\d+)s and t=(\d+\.\d+)s'

# Open the input file and parse lines
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)

    # Write header
    writer.writerow(['speaker', 'start', 'end'])

    for line in infile:
        line = line.strip()
        if line:
            match = re.match(pattern, line)
            if match:
                speaker, start, end = match.groups()
                writer.writerow([speaker, float(start), float(end)])

print(f"Parsed diarization data from {input_file} to {output_file}")
