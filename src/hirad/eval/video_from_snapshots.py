import cv2
import numpy as np
from PIL import Image
import os
import re
from datetime import datetime
import glob
from pathlib import Path
from tqdm import tqdm
import subprocess
import shutil

def parse_filename(filename):
    """Parse filename to extract date, hour, type, and member"""
    # Pattern for prediction files: YYYYMMDD-HH-tp-prediction_MM.png
    pred_pattern = r'(\d{8})-(\d+)-tp-prediction_(\d+)\.png'
    # Pattern for target/mean files: YYYYMMDD-HH-tp-(target|mean-prediction).png
    other_pattern = r'(\d{8})-(\d+)-tp-(target|mean-prediction)\.png'
    
    pred_match = re.match(pred_pattern, filename)
    if pred_match:
        date, hour, member = pred_match.groups()
        return date, hour, 'prediction', int(member)
    
    other_match = re.match(other_pattern, filename)
    if other_match:
        date, hour, img_type = other_match.groups()
        return date, hour, img_type, None
    
    return None

def get_sorted_timestamps(image_folder):
    """Get all unique timestamps sorted chronologically"""
    timestamps = set()
    for filename in os.listdir(image_folder):
        parsed = parse_filename(filename)
        if parsed:
            date, hour, _, _ = parsed
            timestamps.add((date, hour))
    
    return sorted(list(timestamps))

def load_and_resize_image(filepath, target_size):
    """Load image and resize to target size, converting RGBA to RGB"""
    if not os.path.exists(filepath):
        # Create blank image if file doesn't exist
        return np.zeros((*target_size[::-1], 3), dtype=np.uint8)
    
    img = Image.open(filepath)
    
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img)

def create_grid_layout(images, grid_shape):
    """Create a grid layout from list of images"""
    rows, cols = grid_shape
    if len(images) != rows * cols:
        # Pad with blank images if needed
        while len(images) < rows * cols:
            images.append(np.zeros_like(images[0]))
    
    # Arrange images in grid
    image_rows = []
    for i in range(rows):
        row_images = images[i*cols:(i+1)*cols]
        image_rows.append(np.hstack(row_images))
    
    return np.vstack(image_rows)


def create_video_from_existing_images(image_folder, output_path, layout_type="all_members", fps=12):
    """Create video directly from existing images without creating intermediate frames"""
    timestamps = get_sorted_timestamps(image_folder)
    
    if not timestamps:
        print("No valid images found!")
        return
    
    # Get image size
    sample_file = next(Path(image_folder).glob("*.png"))
    sample_img = Image.open(sample_file)
    img_width, img_height = sample_img.size
    if layout_type == "all_members":
        img_width //= 8
        img_height //= 8
    
    print(f"Sample image size: {img_width} x {img_height}")
    
    # Create one test frame to get EXACT dimensions
    if layout_type == "all_members":
        # Load real images for the first timestamp to get exact dimensions
        first_date, first_hour = timestamps[0]
        
        # Load 16 prediction images
        pred_images = []
        for member in range(16):
            filename = f"{first_date}-{first_hour}-tp-prediction_{member}.png"
            filepath = Path(image_folder) / filename
            img = load_and_resize_image(str(filepath), (img_width, img_height))
            pred_images.append(img)
        
        pred_grid = create_grid_layout(pred_images, (4, 4))
        
        # Load target and mean and baseline
        target_file = f"{first_date}-{first_hour}-tp-target.png"
        target_img = load_and_resize_image(str(Path(image_folder) / target_file), (img_width, img_height))
        
        mean_file = f"{first_date}-{first_hour}-tp-mean-prediction.png"
        mean_img = load_and_resize_image(str(Path(image_folder) / mean_file), (img_width, img_height))

        baseline_file = f"{first_date}-{first_hour}-tp-baseline.png"
        baseline_img = load_and_resize_image(str(Path(image_folder) / baseline_file), (img_width, img_height))

        bottom_row = np.hstack([baseline_img, mean_img, target_img])
        
        # Pad if needed
        pad_width = pred_grid.shape[1] - bottom_row.shape[1]
        if pad_width > 0:
            padding = np.zeros((bottom_row.shape[0], pad_width, 3), dtype=np.uint8)
            bottom_row = np.hstack([bottom_row, padding])
        
        test_frame = np.vstack([pred_grid, bottom_row])
        frame_height, frame_width = test_frame.shape[:2]  # numpy: (height, width)
        
    else:  # single_member
        frame_width = 4 * img_width
        frame_height = img_height
    
    # print(f"Frame dimensions (H x W): {frame_height} x {frame_width}")
    
    # Initialize VideoWriter with CORRECT dimension order (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))  # OpenCV: (width, height)
    
    if not out.isOpened():
        print("Failed to initialize VideoWriter!")
        return False
    
    # print(f"VideoWriter initialized with (W x H): {frame_width} x {frame_height}")
    
    # Create video
    for date, hour in tqdm(timestamps, desc=f"Creating {layout_type} video"):
        
        if layout_type == "all_members":
            # Load 16 prediction images
            pred_images = []
            for member in range(16):
                filename = f"{date}-{hour}-tp-prediction_{member}.png"
                filepath = Path(image_folder) / filename
                img = load_and_resize_image(str(filepath), (img_width, img_height))
                pred_images.append(img)
            
            pred_grid = create_grid_layout(pred_images, (4, 4))
            
            # Load target and mean
            target_file = f"{date}-{hour}-tp-target.png"
            target_img = load_and_resize_image(str(Path(image_folder) / target_file), (img_width, img_height))
            
            mean_file = f"{date}-{hour}-tp-mean-prediction.png"
            mean_img = load_and_resize_image(str(Path(image_folder) / mean_file), (img_width, img_height))

            baseline_file = f"{date}-{hour}-tp-baseline.png"
            baseline_img = load_and_resize_image(str(Path(image_folder) / baseline_file), (img_width, img_height))
            
            bottom_row = np.hstack([baseline_img, mean_img, target_img])
            
            # Pad if needed (using same logic as test frame)
            pad_width = pred_grid.shape[1] - bottom_row.shape[1]
            if pad_width > 0:
                padding = np.zeros((bottom_row.shape[0], pad_width, 3), dtype=np.uint8)
                bottom_row = np.hstack([bottom_row, padding])
            
            frame = np.vstack([pred_grid, bottom_row])
            
        else:  # single_member
            pred_file = f"{date}-{hour}-tp-prediction_15.png"
            pred_img = load_and_resize_image(str(Path(image_folder) / pred_file), (img_width, img_height))
            
            target_file = f"{date}-{hour}-tp-target.png"
            target_img = load_and_resize_image(str(Path(image_folder) / target_file), (img_width, img_height))
            
            mean_file = f"{date}-{hour}-tp-mean-prediction.png"
            mean_img = load_and_resize_image(str(Path(image_folder) / mean_file), (img_width, img_height))

            baseline_file = f"{date}-{hour}-tp-baseline.png"
            baseline_img = load_and_resize_image(str(Path(image_folder) / baseline_file), (img_width, img_height))
            
            frame = np.hstack([baseline_img, mean_img, pred_img, target_img])
        
        # Verify frame dimensions EXACTLY match VideoWriter
        actual_height, actual_width = frame.shape[:2]
        if actual_width != frame_width or actual_height != frame_height:
            print(f"ERROR: Frame size mismatch!")
            print(f"Expected: {frame_width} x {frame_height}")
            print(f"Got: {actual_width} x {actual_height}")
            print(f"Resizing frame to match...")
            # Force resize to exact dimensions
            frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Ensure 3 channels
        if frame.shape[2] != 3:
            print(f"ERROR: Frame has {frame.shape[2]} channels, expected 3")
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]  # Drop alpha channel

        # print(f"Frame shape: {frame.shape}")
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Write frame
        success = out.write(frame_bgr)
        # if not success:
        #     print(f"Failed to write frame for {date}-{hour}")
        #     break
    
    out.release()
    print(f"Video saved: {output_path}")
    return True


def main():
    image_folder = Path("/capstor/scratch/cscs/pstamenk/outputs/generation/generate_8_attention/tp")
    
    # Create both videos directly from existing images
    create_video_from_existing_images(
        image_folder, 
        image_folder / "precipitation_all_members.avi", 
        layout_type="all_members"
    )
    
    create_video_from_existing_images(
        image_folder, 
        image_folder / "precipitation_single_member.avi", 
        layout_type="single_member"
    )

if __name__ == "__main__":
    main()