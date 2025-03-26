# Clone GitHub repo
# !git clone https://github.com/dylandru/BaseballCV.git

# Set working directory

from ultralytics import YOLO
from mmpose.apis import MMPoseInferencer
import cv2
import pandas as pd
import random
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

### Track pitcher mechanics

# Detection model trained and pulled from BaseballCV repo
detection_model = YOLO(model="pitcher_hitter_catcher_detector_v4.pt") 

# RTMPose model suggested in article
pose_model = MMPoseInferencer(pose2d="rtmpose-m_8xb256-420e_coco-256x192")

# Keypoint labels
labels = {
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

# Function to obtain largest frame
def get_frame(bboxes):
    # Initialize the values with the first bounding box
    max_x = max_y = float('-inf')
    min_x = min_y = float('inf')
    
    # Get frame size
    for box in bboxes:
        try:
            x1, y1, x2, y2 = box[0]
                
            # Update the maximum upper-left coordinates
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            
            # Update the minimum lower-right coordinates
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
        
        except IndexError:
            continue
        
    # Return the final combined bounding box
    return [round(min_x), round(min_y), round(max_x), round(max_y)]

# Calculate bbox area function
def calculate_area(coords):
    x1, y1, x2, y2 = coords
    width = x2 - x1
    height = y2 - y1
    area = width * height
    return area

# Define pose data collection function
def get_pose_data(video, show_detection: bool = False):
    # Boolean safeguard for showing PHC detection video
    if not isinstance(show_detection, bool):
        raise ValueError("Argument must be a boolean.")
    
    # Actual function
    pit_detect = detection_model.predict(video, show=show_detection)
    
    frames = []

    for i in range(len(pit_detect)):
        boxes = pit_detect[i].boxes
        frames.append(boxes.xyxy[(boxes.cls == 1).nonzero(as_tuple=True)[0]].tolist())
        
    x1, y1, x2, y2 = get_frame(frames)
    
    # Establish frame number
    frame_no = 0

    pose_results = []

    # Open the video file
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        while True:
            # Read the next frame from the video
            ret, frame = cap.read()
            
            if not ret:
                break  # Exit if no more frames are available
            
            if not frames[frame_no]:
                frame_no += 1
                continue
            
            # Crop the frame using the xyxy coordinates
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Run pose model on cropped frame
            result_generator = pose_model(cropped_frame)
            result = next(result_generator)
            pose_results.append(result)
            
            frame_no += 1
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    
    data = []
    
    for index in range(len(pose_results)):
        # Extract pose data for pitcher (take the lowest upper y value from the bounding box for each detected pose)
        pitcher_pose = max(pose_results[index]['predictions'][0], key=lambda person: calculate_area(person['bbox'][0]))
        
        row = {}
        for label, body_part in labels.items():
            # Extract the x, y, and confidence values for the given label
            try:
                row[f'{body_part}_x'] = pitcher_pose['keypoints'][label][0]
                row[f'{body_part}_y'] = pitcher_pose['keypoints'][label][1]
                row[f'{body_part}_confidence'] = pitcher_pose['keypoint_scores'][label]
            except (IndexError, KeyError, AttributeError):
                # NaN for missing/malformed data
                row[f'{body_part}_x'] = None
                row[f'{body_part}_y'] = None
                row[f'{body_part}_confidence'] = None
        data.append(row)

    df = pd.DataFrame(data)
    
    df = df.assign(x1=x1, y1=y1, x2=x2, y2=y2)
    
    return df

# Read in pitch IDs
all_pitch_ids = pd.read_csv('mlb_play_ids.csv')

# Comparing pitchers to Giolito (Bieber, Crouse, Ty. Rogers)
pitcher_ids = [608337, 669456, 668968, 643511]

# Select random subset of pitches for analysis
filtered_ids = (
    all_pitch_ids[
        (all_pitch_ids['matchup.pitcher.id'].isin(pitcher_ids)) &
        # Ensure all pitches from same stadium
        (all_pitch_ids['home_team'] == 'Los Angeles Angels')    
    ]
    .groupby('matchup.pitcher.id', group_keys=False)
    .apply(lambda x: x.sample(n=10, random_state=218))
)

# Directory to save videos
video_dir = 'videos'
os.makedirs(video_dir, exist_ok=True)

# Dictionary to store results
pose_data_dict = {}

# Initialize the lock
lock = Lock()

# Run in parallel
def process_play_id(play_id, pitcher_id, pitch_type, at_bat_num, pitch_num):
    driver = webdriver.Firefox()
    try:
        url = f'https://baseballsavant.mlb.com/sporty-videos?playId={play_id}'
        driver.get(url)

        # Wait for the video element to load and fetch the MP4 URL
        video_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "video"))
        )
        source_element = video_element.find_element(By.TAG_NAME, "source")
        video_url = source_element.get_attribute("src")

        # Download the video
        video_path = os.path.join(video_dir, f"{play_id}.mp4")
        response = requests.get(video_url, stream=True)
        if response.status_code == 200:
            with open(video_path, "wb") as video_file:
                for chunk in response.iter_content(chunk_size=1024):
                    video_file.write(chunk)
        else:
            raise Exception(f"Failed to download video: {response.status_code}")

        # Run pose estimation
        pose_data = get_pose_data(video_path)
        
        # Use the lock to synchronize access to pose_data_dict
        with lock:
            pose_data_dict[(play_id, pitcher_id, pitch_type, at_bat_num, pitch_num)] = pose_data

    except Exception as e:
        print(f"Error processing playId {play_id}: {e}")
    finally:
        driver.quit()

# Use ThreadPoolExecutor to process play IDs in parallel
with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on system capacity
    executor.map(lambda x: process_play_id(x[0], x[1], x[2], x[3], x[4]), 
                 zip(filtered_ids['playId'], filtered_ids['matchup.pitcher.id'], filtered_ids['details.type.description'], filtered_ids['atBatIndex'], filtered_ids['pitchNumber']))

# Save the results dictionary as a pickle file
with open('pitcher_comps.pkl', 'wb') as f:
    pickle.dump(pose_data_dict, f)



### Perform DTW analysis
# Load in data
with open('pitcher_comps.pkl', 'rb') as f:
    datasets = pickle.load(f)
    
# Establish throwing hand
throws = "right"
knee_id = "left" if throws == "right" else "right"

filtered_datasets = {}
keys = list(datasets.keys())

# To measure arm action
columns_to_use = arm_keypoints

# To measure base
# columns_to_use = base_keypoints

# Manual data cleaning
for i in range(len(keys)):
    current_dataset = datasets[keys[i]]
    
    check = current_dataset.loc[current_dataset[f'{knee_id}_knee_y'].idxmin():]
    check = check.loc[:current_dataset[f'{throws}_ankle_y'].idxmin()]
    keypoint_columns = [col for col in current_dataset.columns if '_x' in col or '_y' in col]
    displacement = np.sqrt(((check[keypoint_columns] - check[keypoint_columns].shift(1)) ** 2).sum(axis=1))

    # Filter dataset to create comprable sequences
    if 50 <= len(check) <= 100 and (displacement <= 500).all():
        current_dataset = check
    else:
        displacement = np.sqrt(((current_dataset[keypoint_columns] - current_dataset[keypoint_columns].shift(1)) ** 2).sum(axis=1))
        threshold = displacement.mean() + 3 * displacement.std()
        if (displacement.iloc[:100] <= threshold).all():
            # Remove potential camera cut or mislabels following pitch
            for _ in range(len(displacement)):
                for idx, value in displacement.items():
                    if value > threshold:
                        current_dataset = current_dataset.drop(index=idx).reset_index(drop=True)
                        displacement = np.sqrt(((current_dataset[keypoint_columns] - current_dataset[keypoint_columns].shift(1)) ** 2).sum(axis=1))
                        # threshold = displacement.mean() + 2.5 * displacement.std()
                        break
                else:
                    break
            # Determine start of pitching motion (peak leg lift)
            current_dataset = current_dataset.loc[current_dataset[f'{knee_id}_knee_y'].idxmin():]
            # Determine end of pitching motion
            current_dataset = current_dataset.loc[:current_dataset[f'{throws}_ankle_y'].idxmin()]
            
            # Truncate longer datasets
            if len(current_dataset) > 100:
                displacement = np.sqrt(((current_dataset[keypoint_columns] - current_dataset[keypoint_columns].shift(1)) ** 2).sum(axis=1))
                threshold = displacement.mean() + 5 * displacement.std()
                # Remove potential camera cut or mislabels following pitch
                for _ in range(len(displacement)):
                    for idx, value in displacement.items():
                        if value > threshold:
                            current_dataset = current_dataset.drop(index=idx).reset_index(drop=True)
                            displacement = np.sqrt(((current_dataset[keypoint_columns] - current_dataset[keypoint_columns].shift(1)) ** 2).sum(axis=1))
                            # threshold = displacement.mean() + 2.5 * displacement.std()
                            break
                    else:
                        break
                # Determine start of pitching motion (peak leg lift)
                current_dataset = current_dataset.loc[current_dataset[f'{knee_id}_knee_y'].idxmin():]
                # Determine end of pitching motion
                current_dataset = current_dataset.loc[:current_dataset[f'{throws}_ankle_y'].idxmin()]
        else: # For when camera began from a different angle than traditional broadcast view
            # Determine when camera returns to traditional broadcast view
            camera_cut = displacement.iloc[:100][displacement.iloc[:100] > threshold].last_valid_index()
            current_dataset = current_dataset.loc[camera_cut:].reset_index(drop=True)
            displacement = np.sqrt(((current_dataset[keypoint_columns] - current_dataset[keypoint_columns].shift(1)) ** 2).sum(axis=1))
            # Remove potential camera cut or mislabels following pitch #
            for _ in range(len(displacement)):
                for idx, value in displacement.items():
                    if value > threshold:
                        current_dataset = current_dataset.drop(index=idx).reset_index(drop=True)
                        displacement = np.sqrt(((current_dataset[keypoint_columns] - current_dataset[keypoint_columns].shift(1)) ** 2).sum(axis=1))
                        break
                else:
                    break
            # Determine start of pitching motion (peak leg lift)
            current_dataset = current_dataset.loc[current_dataset[f'{knee_id}_knee_y'].idxmin():]
            # Determine end of pitching motion
            current_dataset = current_dataset.loc[:current_dataset[f'{throws}_ankle_y'].idxmin()]

            # Truncate longer datasets
            if len(current_dataset) > 100:
                displacement = np.sqrt(((current_dataset[keypoint_columns] - current_dataset[keypoint_columns].shift(1)) ** 2).sum(axis=1))
                threshold = displacement.mean() + 5 * displacement.std()
                # Remove potential camera cut or mislabels following pitch
                for _ in range(len(displacement)):
                    for idx, value in displacement.items():
                        if value > threshold:
                            current_dataset = current_dataset.drop(index=idx).reset_index(drop=True)
                            displacement = np.sqrt(((current_dataset[keypoint_columns] - current_dataset[keypoint_columns].shift(1)) ** 2).sum(axis=1))
                            # threshold = displacement.mean() + 2.5 * displacement.std()
                            break
                    else:
                        break
                # Determine start of pitching motion (peak leg lift)
                current_dataset = current_dataset.loc[current_dataset[f'{knee_id}_knee_y'].idxmin():]
                # Determine end of pitching motion
                current_dataset = current_dataset.loc[:current_dataset[f'{throws}_ankle_y'].idxmin()]
    
    # Find differences from previous points
    for col in current_dataset.columns:
        if not col.endswith(('_confidence', '1', '2')):
            diff_col = f"{col}_diff"
            
            current_dataset[diff_col] = current_dataset[col].diff()
            
            current_dataset[diff_col] = current_dataset[diff_col].fillna(0)
            
    filtered_datasets[keys[i]] = np.column_stack([current_dataset[col].values for col in columns_to_use])

# Establish reference pitcher data
ref_pitcher = 608337
ref_data = {key: value for key, value in filtered_datasets.items() if key[1] == ref_pitcher}
ref_keys = list(ref_data.keys())

# Establish test pitchers data
comp_ids = [669456, 668968, 643511] # Bieber, Crouse, Rogers
test_pitcher = comp_ids[2]
test_data = {key: value for key, value in filtered_datasets.items() if key[1] == test_pitcher}
test_keys = list(test_data.keys())

# Establish datasets for comparison
ref_dataset = filtered_datasets[ref_keys[7]]
test_dataset = filtered_datasets[test_keys[0]]

# Compute the Multivariate DTW distance
distance, path = fastdtw(ref_dataset, test_dataset, dist=euclidean)

# Plot the warping path
plt.subplot(2, 1, 2)
arm1_indices = [i for i, _ in path]
arm2_indices = [j for _, j in path]
plt.plot(arm1_indices, arm2_indices, label="Warping Path", color='red', marker='.')
plt.plot([0, max(arm1_indices)], [0, max(arm2_indices)], label="Perfect Alignment", color='black', linestyle='--')
plt.title("Giolito (ref) vs Rogers (test) Warping Path")
plt.xlabel("Ref Index")
plt.ylabel("Test Index")
plt.grid(True)
plt.legend()
plt.gca().spines['top'].set_color('white')
plt.gca().spines['right'].set_color('white')
plt.gca().spines['left'].set_color('white')
plt.gca().spines['bottom'].set_color('white')

# Plotting the sequences
aligned_ref = [ref_dataset[i] for i, _ in path]
aligned_test = [test_dataset[j] for _, j in path]

# Plotting the sequences
plt.figure(figsize=(12, 12))

# Plot aligned subsequences with proper labels from ref_names
for idx, col in enumerate(columns_to_use):
    plt.subplot(len(columns_to_use), 1, idx+1)
    
    col_index = columns_to_use.index(col)
    
    # Plot aligned ref subsequence
    plt.plot(np.arange(1, 1 + len(aligned_ref)), 
             [s[col_index] for s in aligned_ref], label=f"Ref", marker='o', color='b')
    
    # Plot aligned test subsequence
    plt.plot(np.arange(1, 1 + len(aligned_test)), 
             [s[col_index] for s in aligned_test], label=f"Test", marker='x', color='r')
    
    # Title and labels
    plt.title(f"Aligned Sequence for {col}")
    plt.xlabel("Time index")
    plt.ylabel("Value")
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))

plt.tight_layout()
plt.suptitle("Giolito (ref) vs Rogers (test) Aligned Arm Action Comparison", fontsize=16, y=1.02)
plt.show()