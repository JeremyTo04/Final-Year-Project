
import pandas as pd

def encode_action_units(csv_file_path, output_file_path):
    # Step 1: Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Step 2: Filter the DataFrame to include only the relevant columns
    # relevant_columns = ['Action Units', 'Estimated Emotion (7 class)', 'glasses', 'fringe', 'gender'] # casme2
    # relevant_columns = ['Action Units', "Ethnicity", 'glasses', "fringe", "gender"] # sammlong
    relevant_columns = ['Action Units', 'Estimated Emotion (7 class)', 'age', 'gender', 'ethnicity', 'glasses', 'fringe', 'Predicted'] # 4dme
    df = df[relevant_columns]

    # # Step 3: Normalize the Estimated Emotion column to handle case sensitivity and spaces
    # df['Estimated Emotion (7 class)'] = df['Estimated Emotion (7 class)'].str.lower().str.strip()

    # # Step 4: Map Estimated Emotion to numerical values (1-7)
    emotion_mapping = {
        'happiness': 1,
        'disgust': 2,
        'repression': 3,
        'surprise': 4,
        'fear': 5,
        'sadness': 6,
        'others': 7
    }
    df['Estimated Emotion (7 class)'] = df['Estimated Emotion (7 class)'].map(emotion_mapping)

    ethnicity_mapping = {
        "middle eastern": 1,
        "white": 2,
        "asian": 3,
        "latino hispanic": 4,
        "black": 5,
        "indian": 6
    }

    df["ethnicity"] = df["ethnicity"].map(ethnicity_mapping)

    # Step 5: Convert Yes/No columns to 0/1
    binary_mapping = {'yes': 1, 'no': 0}
    df['glasses'] = df['glasses'].str.lower().str.strip().map(binary_mapping)
    df['fringe'] = df['fringe'].str.lower().str.strip().map(binary_mapping)

    binary_mapping = {'female': 1, 'male': 0}
    df['gender'] = df['gender'].str.lower().str.strip().map(binary_mapping)

    # Step 6: Identify all unique action units and their combinations
    unique_combinations = set(df['Action Units'])
    
    # Step 7: Map each unique combination to a unique numerical value
    combination_mapping = {combination: i for i, combination in enumerate(unique_combinations, start=1)}

    # Step 8: Apply the mapping to the Action Units column
    df['Action Units'] = df['Action Units'].map(combination_mapping)

    # print the mapping
    print("Action Unit Combination Mapping:")
    for combination, numeric_value in combination_mapping.items():
        print(f"{combination}: {numeric_value}")

    # # for emotions:
    # unique_combinations = set(df['Emotion'])
    
    # # Step 7: Map each unique combination to a unique numerical value
    # combination_mapping = {combination: i for i, combination in enumerate(unique_combinations, start=1)}

    # # Step 8: Apply the mapping to the Action Units column
    # df['Emotion'] = df['Emotion'].map(combination_mapping)

    # # print the mapping
    print("Action Unit Combination Mapping:")
    for combination, numeric_value in combination_mapping.items():
        print(f"{combination}: {numeric_value}")

    # Step 9: Save the updated DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)
    print(f"Encoded and filtered data have been saved to {output_file_path}")


def merge_dataset():
    # Load the CSV files
    predicted_emotion_df = pd.read_csv(r'C:\Users\jeret\OneDrive\Documents\GitHub\Final-Year-Project\MMNet-main\predicted_emotion.csv')
    casme2_df = pd.read_csv(r'C:\Users\jeret\OneDrive\Desktop\fit 3163\Actual non assignment stuff\deepface annotations\final annotations\CASME2-coding-20140508.csv')

    # Select only the necessary columns from CASME2-coding-20140508.csv
    casme2_df_selected = casme2_df[['Subject', 'Filename', 'Action Units', 'Estimated Emotion (7 class)', 'glasses', 'fringe', 'gender', 'ethnicity', 'age']]

    # Merge the predicted_emotion_df with casme2_df_selected on 'subject' and 'Image'
    merged_df = pd.merge(predicted_emotion_df, casme2_df_selected, on=['Subject', 'Filename'], how='left')

    # Save the result back to a CSV file
    merged_df.to_csv('updated_predicted_emotion.csv', index=False)

    print("Data merged and saved to updated_predicted_emotion.csv")



if __name__ == '__main__':
    # merge_dataset()
    # Example usage:
    # input_csv = r"C:\Users\jeret\OneDrive\Desktop\fit 3163\Actual non assignment stuff\Datasets\CASME2\CASME2\CASME2-coding-20140508_COPY.csv"  # Replace with your input CSV file path
    # input_csv = r"C:\Users\jeret\OneDrive\Desktop\fit 3163\Actual non assignment stuff\sammlong (Alyssa).csv"
    input_csv = r'C:\Users\jeret\OneDrive\Documents\GitHub\Final-Year-Project\updated_predicted_emotion.csv'
    output_csv = 'final_microexpression_data_CASME2.csv'  # Replace with your desired output CSV file path
    encode_action_units(input_csv, output_csv)
    # encode_action_units(input_csv, output_csv)



"""Other stuff"""

    # # Step 6: Identify all unique action units and map them to numerical values
    # unique_aus = set()
    # for aus in df['Action Units']:
    #     for au in aus.split('+'):
    #         unique_aus.add(au.strip())

    # au_mapping = {au: i for i, au in enumerate(unique_aus, start=1)}

    # # Step 7: Create a column for all the unique combinations of action units and map them to numerical values
    # df['Action Units'] = df['Action Units'].apply(lambda x: [au_mapping[au.strip()] for au in x.split('+')])
    # df['Action Units'] = df['Action Units'].apply(lambda x: '+'.join(map(str, x)))


    # # Step 6: Identify all unique action units in the dataset
    # unique_aus = set()
    # for aus in df['Action Units']:
    #     for au in aus.split('+'):
    #         unique_aus.add(au.strip())

    # # Step 7: Create a column for each unique action unit
    # for au in unique_aus:
    #     df[f"AU_{au}"] = df['Action Units'].apply(lambda x: 1 if au in x.split('+') else 0)

    # # Step 8: Drop the 'Action Units' column
    # df = df.drop(columns=['Action Units'])