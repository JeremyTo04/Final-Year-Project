import os

# Define the base directory
base_dir = "C:\\Users\\alyss\\OneDrive\\Documents\\GitHub\\Final-Year-Project\\STSTNet-master\\input"

# Walk through all directories and files within the base directory
for root, dirs, files in os.walk(base_dir):
    for filename in files:
        # Check if the filename ends with .png and has a trailing space
        if filename.endswith('.png') and filename[-5] == ' ':
            # Remove the trailing space
            new_filename = filename[:-5] + '.png'
            old_file_path = os.path.join(root, filename)
            new_file_path = os.path.join(root, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
           # print(f"Renamed: {old_file_path} -> {new_file_path}")
