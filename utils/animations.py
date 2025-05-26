import cv2
import os
from tqdm import tqdm

def video_from_folder(folder_path, fps=30):
    
    output_video_path = 'train.mp4'
    
    images = [f'{i:04}.png' for i,_ in enumerate(os.listdir(folder_path))]

    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get the width and height for the video
    frame = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = frame.shape

    # Set the video writer (output video settings)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image in tqdm(images):
        img_path = os.path.join(folder_path, image)
        img = cv2.imread(img_path)
        video_writer.write(img)

    video_writer.release()

    print(f"Video created successfully and saved to {output_video_path}.")