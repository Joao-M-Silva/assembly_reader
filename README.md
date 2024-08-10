# Overview
This repository contains an AI powered implementation to interpret assemblage videos.

# Usage
1. Make sure you have ultralytics installed in your environment
   ```
   pip install ultralytics==8.2.74
   ```
   or
   ```
   pip install -r requirements.txt
   ```

2. In order to process a source video run
   ```
   python main.py --file_path {OUTPUT_VIDEO_PATH} --video_path {SOURCE_VIDEO_PATH}
   ```
   or a specific number of frames
   ```
   python main.py --file_path {OUTPUT_VIDEO_PATH} --video_path {SOURCE_VIDEO_PATH} -N {NUMBER_FRAMES}
   ```

# Results
![Design sem nome (4)](https://github.com/user-attachments/assets/966a1e12-2997-4cf0-b319-67fea14999fa)


