for dir in output / * /; do
# 1. Define source and destination paths
# ${dir%/} removes the trailing slash from the directory name
video_path = "${dir%/}"
src_folder = "$video_path/filtered_segments"
dest_folder = "$video_path/filtered_segments_wav"

# 2. Check if the source folder exists before proceeding
if [-d "$src_folder"]; then
echo
"Processing: $video_path"

# 3. Create the new directory
mkdir - p
"$dest_folder"

# 4. Loop through all .m4a files and convert them
for file in "$src_folder" / *.m4a; do
# Check if file exists to handle empty directories
[-e "$file"] | |
continue

# Get the filename without the path and extension
filename =$(basename "$file".m4a)

# Convert to wav using ffmpeg
# -i: input file
# -n: do not overwrite if output already exists
# -loglevel error: hides technical logs so you only see errors
ffmpeg - n - loglevel
error - i
"$file" "$dest_folder/$filename.wav"
done
fi
done
echo
"Conversion complete."