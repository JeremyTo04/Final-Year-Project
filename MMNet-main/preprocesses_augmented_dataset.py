import os
from PIL import Image

# Function to process images
def process_images(image_directory, output_directory):
    # List all PNG files in the given directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]
    
    for image_file in image_files:
        # Split the filename
        parts = image_file.split('_')
        
        # Ensure the filename has the expected number of parts
        if len(parts) < 3:
            print(f"Skipping '{image_file}': not enough parts in filename.")
            continue
        
        # Extract relevant parts
        A = parts[0]  # 'onset', 'apex', or 'offset'
        frame_info = '_'.join(parts[2:-2])  # b - f (everything between A and the last two parts)
        frame_number = parts[-1]  # Last part is frame number
        subject_folder_name = parts[1]  # Subject folder name (section B)

        # Create the full output path for the subject folder
        subject_folder_path = os.path.join(output_directory, subject_folder_name, f"{frame_info}_aug")

        # Create the folder if it doesn't exist
        os.makedirs(subject_folder_path, exist_ok=True)

        # Determine the new image name based on A
        if A == 'onset':  # Onset
            new_image_name = f"Frame_000000001.jpg"
        elif A == 'apex':  # Apex
            new_image_name = f"Frame_000000002.jpg"
        elif A == 'offset':  # Offset
            new_image_name = f"Frame_000000003.jpg"
        else:
            print(f"Skipping '{image_file}': A value not recognized.")
            continue

        # Full paths
        old_image_path = os.path.join(image_directory, image_file)
        new_image_path = os.path.join(subject_folder_path, new_image_name)

        # Open the PNG image and convert to JPG
        try:
            with Image.open(old_image_path) as img:
                # Convert to RGB (PNG might have an alpha channel)
                img = img.convert('RGB')
                # Save as JPG
                img.save(new_image_path, 'JPEG')
            print(f"Converted and moved '{old_image_path}' to '{new_image_path}'")
        except Exception as e:
            print(f"Error processing '{old_image_path}': {e}")

# Set the paths
image_directory = r"C:\Users\jeret\Downloads\young_manipulated_onset_images" # Change this to your actual input directory
output_directory = r"C:\Users\jeret\OneDrive\Documents\GitHub\Final-Year-Project\MMNet-main\4dme\micro_short_gray_video\micro short gray video\micro short gray video"  # Change this to your actual output directory

# Call the function
process_images(image_directory, output_directory)

print("Image processing complete.")