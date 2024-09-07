import os
import pandas as pd

def rename_files_remove_whitespace(base_dir):
    """
    Renames files in the specified directory to remove trailing whitespace before the '.png' extension.
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
    """
    all_data = []
    for subject_folder in os.listdir(root_directory):
        subject_path = os.path.join(root_directory, subject_folder)
        if os.path.isdir(subject_path):
            csv_file_path = os.path.join(subject_path, 'predicted_emotions.csv')
            if os.path.exists(csv_file_path):
                # skip the header row
                df = pd.read_csv(csv_file_path, skiprows=1, header=None, names=['ImageFile', 'PredictedEmotion'])
                all_data.append(df)
    compiled_df = pd.concat(all_data, ignore_index=True)
    output_file_path = os.path.join(root_directory, 'compiled_emotions.csv')
    compiled_df.to_csv(output_file_path, index=False)

    print(f"Compiled CSV file saved to: {output_file_path}")
    
def add_emotion_label(csv_file_path):
    df = pd.read_csv(csv_file_path)
    emotion_mapping = {0: 'negative', 1: 'positive', 2: 'surprise'}
    df['emotion label'] = df['Predicted Emotion'].map(emotion_mapping)
    df.to_csv(csv_file_path, index=False)
    return df

def add_action_unit_column(emotions_csv_path, action_units_xlsx_path, output_csv_path):
    emotions_df = pd.read_csv(emotions_csv_path)
    action_units_df = pd.read_excel(action_units_xlsx_path)
    merged_df = pd.merge(emotions_df, action_units_df[['File name', 'Action Units']], on='File name', how='left')
    merged_df.to_csv(output_csv_path, index=False)
    print(f"New CSV file with 'Action unit' column created at: {output_csv_path}")
    
def assign_unique_ids(csv_path, output_csv_path):
    df = pd.read_csv(csv_path)
    df['Estimated Emotion ID'] = df['Estimated Emotion (7 class)'].astype('category').cat.codes + 1
    df.to_csv(output_csv_path, index=False)
    print(f"New CSV file with 'Estimated Emotion ID' column created at: {output_csv_path}")


def match_emotions(predicted_emotions_path, new_casme_path, output_path):
    # Read the CSV files into DataFrames
    predicted_df = pd.read_csv(predicted_emotions_path)
    new_casme_df = pd.read_csv(new_casme_path)

    # Merge the DataFrames on 'Filename' column, allowing for duplicates
    merged_df = pd.merge(new_casme_df, predicted_df[['Filename', 'Predicted Emotion']], on='Filename', how='left')
    
    # Drop rows where 'Predicted Emotion' is NaN
    merged_df = merged_df.dropna(subset=['Predicted Emotion'])

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_path, index=False)

    return merged_df

def replace_values(input_path, output_path):
    df = pd.read_csv(input_path)
    
    if 'gender' in df.columns:
        # Replace 'Female' with 0 and 'Male' with 1
        df['gender'] = df['gender'].replace({'Female': 0, 'Male': 1})
        df['gender'] = df['gender'].astype(int)
    else:
        print("The 'gender' column is not found in the CSV file.")
    
    df.to_csv(output_path, index=False)
    
    return df



if __name__ == "__main__":

    csv_path=r"C:\Users\alyss\Downloads\1new_CASME2-coding-20140508.csv"
    output_csv_path=r"C:\Users\alyss\Downloads\1new_CASME2-coding-20140508.csv"
    assign_unique_ids(csv_path, output_csv_path)