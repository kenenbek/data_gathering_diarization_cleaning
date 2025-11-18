"""
Split CSV file into 4 equal parts for parallel processing.
"""

import pandas as pd
import math

# Read the original CSV
input_csv = "youtube_nakta_videos_phase_1.csv"
df = pd.read_csv(input_csv)

print(f"Total videos: {len(df)}")

# Calculate split size
total = len(df)
chunk_size = math.ceil(total / 4)

print(f"Videos per file: ~{chunk_size}")

# Split into 4 parts
for i in range(4):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, total)
    
    chunk_df = df.iloc[start_idx:end_idx]
    
    output_file = f"youtube_nakta_videos_phase_1_part{i+1}.csv"
    chunk_df.to_csv(output_file, index=False)
    
    print(f"✓ Created {output_file} with {len(chunk_df)} videos (rows {start_idx+1}-{end_idx})")

print("\n✓ Split complete!")
print("\nCreated files:")
print("  - youtube_nakta_videos_phase_1_part1.csv")
print("  - youtube_nakta_videos_phase_1_part2.csv")
print("  - youtube_nakta_videos_phase_1_part3.csv")
print("  - youtube_nakta_videos_phase_1_part4.csv")

