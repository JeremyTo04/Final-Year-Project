# import os
# import pandas as pd

# # Define the base directory
# base_dir = "C:\\Users\\alyss\\OneDrive\\Documents\\GitHub\\Final-Year-Project\\STSTNet-master\\input"

# # Walk through all directories and files within the base directory
# for root, dirs, files in os.walk(base_dir):
#     for filename in files:
#         # Check if the filename ends with .png and has a trailing space
#         if filename.endswith('.png') and filename[-5] == ' ':
#             # Remove the trailing space
#             new_filename = filename[:-5] + '.png'
#             old_file_path = os.path.join(root, filename)
#             new_file_path = os.path.join(root, new_filename)
#             # Rename the file
#             os.rename(old_file_path, new_file_path)
#            # print(f"Renamed: {old_file_path} -> {new_file_path}")

# # Define the root directory where the subject folders are located
# root_directory = r"C:\Users\alyss\OneDrive\Documents\GitHub\Final-Year-Project\STSTNet-master\casme2"

# # Initialize an empty list to store dataframes
# all_data = []

# # Traverse through each subject folder in the root directory
# for subject_folder in os.listdir(root_directory):
#     subject_path = os.path.join(root_directory, subject_folder)
    
#     # Check if the path is a directory
#     if os.path.isdir(subject_path):
#         # Define the path to the predicted_emotions.csv file
#         csv_file_path = os.path.join(subject_path, 'predicted_emotions.csv')
        
#         # Check if the CSV file exists in the folder
#         if os.path.exists(csv_file_path):
#             # Read the CSV file into a pandas dataframe, skip the header row
#             df = pd.read_csv(csv_file_path, skiprows=1, header=None, names=['ImageFile', 'PredictedEmotion'])
            
#             # Append the dataframe to the list
#             all_data.append(df)

# # Concatenate all dataframes into one
# compiled_df = pd.concat(all_data, ignore_index=True)

# # Define the output file path for the compiled CSV
# output_file_path = os.path.join(root_directory, 'compiled_emotions.csv')

# # Write the concatenated dataframe to a new CSV file
# compiled_df.to_csv(output_file_path, index=False)

# print(f"Compiled CSV file saved to: {output_file_path}")

import os
import pandas as pd

def rename_files_remove_whitespace(base_dir):
    """
    Renames files in the specified directory to remove trailing whitespace before the '.png' extension.
    
    Args:
        base_dir (str): The base directory where the renaming should occur.
    """
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
                print(f"Renamed: {old_file_path} -> {new_file_path}")

def compile_csv_data(root_directory):
    """
    Compiles data from all 'predicted_emotions.csv' files in subdirectories into one CSV file.
    
    Args:
        root_directory (str): The root directory containing the subject folders with CSV files.
    
    Returns:
        None
    """
    # Initialize an empty list to store dataframes
    all_data = []

    # Traverse through each subject folder in the root directory
    for subject_folder in os.listdir(root_directory):
        subject_path = os.path.join(root_directory, subject_folder)
        
        # Check if the path is a directory
        if os.path.isdir(subject_path):
            # Define the path to the predicted_emotions.csv file
            csv_file_path = os.path.join(subject_path, 'predicted_emotions.csv')
            
            # Check if the CSV file exists in the folder
            if os.path.exists(csv_file_path):
                # Read the CSV file into a pandas dataframe, skip the header row
                df = pd.read_csv(csv_file_path, skiprows=1, header=None, names=['ImageFile', 'PredictedEmotion'])
                
                # Append the dataframe to the list
                all_data.append(df)

    # Concatenate all dataframes into one
    compiled_df = pd.concat(all_data, ignore_index=True)

    # Define the output file path for the compiled CSV
    output_file_path = os.path.join(root_directory, 'compiled_emotions.csv')

    # Write the concatenated dataframe to a new CSV file
    compiled_df.to_csv(output_file_path, index=False)

    print(f"Compiled CSV file saved to: {output_file_path}")

if __name__ == "__main__":

    root_directory = r"C:\Users\alyss\OneDrive\Documents\GitHub\Final-Year-Project\STSTNet-master\casme2"
    compile_csv_data(root_directory)
