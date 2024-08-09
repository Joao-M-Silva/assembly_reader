"""
Process an assembly video 
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

from assembly_reader.assembly_reader import AssemblyReader, Frame


def write_processed_frames(processed_frames: list[Frame], file_path: str):
    if ".mp4" not in file_path:
        file_path += ".mp4"

    out = cv2.VideoWriter(
        file_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        30, 
        processed_frames[0].shape[:2][::-1],
    )

    for frame in processed_frames:
        out.write(np.array(frame.annotate())) 
    out.release()


parser = argparse.ArgumentParser(
    description="Argument for processing the source video",
)

parser.add_argument(
    "--file_path",
    type=str,
    required=True,
    help="Path to save the processed video file",
)
parser.add_argument(
    "--video_path",
    type=str,
    required=True,
    help="Path to the source video",
)
parser.add_argument(
    "--confidence_threshold",
    type=float,
    default=0.4,
    help="Confidence threshold for filtering out bounding box predictions",
)
parser.add_argument(
    "--iou_threshold",
    type=float,
    default=0.7,
    help="""
    IoU threshold used to determine whether two bounding boxes
    corresponde to the same object
    """
)
parser.add_argument(
    "-N",
    "--max_frames",
    type=int,
    default=None,
    help="Maximum number of frames to be processed"
)

args = parser.parse_args()

if __name__ == "__main__":
    reader = AssemblyReader(
        video_path=args.video_path, 
        weights_path="./runs/detect/yolov8n_hands_detector5/weights/best.pt",
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        pen_scratch_detector=YOLO("./runs/detect/yolov8n_scratches_detector2/weights/best.pt")
    )

    processed_frames = reader.process_video(max_frames=args.max_frames)

    write_processed_frames(processed_frames, args.file_path)