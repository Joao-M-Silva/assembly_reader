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

# Example of a processed frame
![example_output](https://github.com/user-attachments/assets/1dd36173-fd66-451e-a10b-3666f5eb1d90)
