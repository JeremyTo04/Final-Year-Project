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

if __name__ == "__main__":

    add_emotion_label(r"C:\Users\alyss\OneDrive\Documents\GitHub\Final-Year-Project\STSTNet-master\casme2\compiled_emotions.csv")
