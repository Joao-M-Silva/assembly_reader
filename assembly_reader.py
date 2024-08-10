"""
Abstraction used for processing assembly videos
retrieving four distinct moments using hand tracking
"""

from copy import deepcopy
from typing import Optional, Literal
from enum import Enum
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import dataclasses
import shapely
import supervision as sv
from ultralytics import YOLO
from ultralytics.engine.results import Results

from operation_counter import (
    TaskClassification,
    ASSEMBLY_ORDER,
    OperationCounter
)


FONT = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 30)
FONT_SMALL = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 20)


class NoHandsDetected(Exception):
    """
    Raised when there is no hands detected on a single frame
    """
    pass

class MoreThan2HandsDetected(Exception):
    """
    Raised when there are more than two hands detected on a single frame
    """
    pass

class UnknownMoment(Exception):
    """
    Raised when none of the four moments are detected
    """
    pass


@dataclasses.dataclass
class BoundingBox:

    """
    Representation of a bounding box prediction

    Attributes:
        x: bounding box center x-coordinate
        y: bounding box center y-coordinate
        w: bounding box width
        h: bounding box height
        conf: bounding box prediction confidence
    """

    x: float
    y: float
    w: float
    h: float
    conf: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """
        Retrives top-left and bottom right coordinates
        """
        return (
            (self.x - (self.w / 2)),
            (self.y - (self.h / 2)),
            (self.x + (self.w / 2)),
            (self.y + (self.h / 2)),
        )

class Hand(BoundingBox):

    """
    Representation of an hand bounding box prediction

    Attributes:
        x: bounding box center x-coordinate
        y: bounding box center y-coordinate
        w: bounding box width
        h: bounding box height
        conf: bounding box prediction confidence
        location: hand location (left/right)
    """

    def __init__(
        self, 
        x: float,
        y: float,
        w: float,
        h: float,
        conf: float,
        location: Optional[Literal["left", "right"]] = None,
    ):
        super().__init__(x, y, w, h, conf)
        self.location = location
    

class ClassificationStatus(Enum):

    """
    Correctness of the current frame classification.
    A task is classified correctly if it matches the operation
    task order
    """

    UNKNOWN = "orange"
    CORRECT = "green"
    WRONG = "red"


class Frame:
    
    """
    Frame task classifier

    Attributes: 
        result: model prediction on a single source video frame
        frame_index: frame position in the source video
        time_per_frame: inverse of the source video FPS
        previous_frame: previous frame in the source video timeline
    """

    def __init__(
        self, 
        result: Results, 
        frame_index: int, 
        time_per_frame: float, 
        previous_frame: "Frame" = None, 
        **kwargs,
    ):
        self.result = result
        self.frame_index = frame_index
        self.time_per_frame = time_per_frame

        self.time = frame_index * time_per_frame

        self.image : np.ndarray = result.orig_img
        self.previous_frame = previous_frame

        # Pen scatch bounding boxes
        if "pen_scratch_detector" in kwargs:
            # Run the specialist model that detects pen scratches
            self.pen_scratches_boxes = kwargs["pen_scratch_detector"](
                self.image, 
                verbose=False,
            )[0].boxes

            # Pen scratches bounding boxes
            self.pen_scratches = []
            for i in range(len(self.pen_scratches_boxes)):
                x, y, w, h = self.pen_scratches_boxes.xywh.numpy()[i]
                conf = self.pen_scratches_boxes.conf[i]
                self.pen_scratches.append(BoundingBox(x, y, w, h, conf))
        else:
            self.pen_scratches_boxes = []

        # Pen scratches bounding boxes
        self.pen_scratches = []
        for i in range(len(self.pen_scratches_boxes)):
            x, y, w, h = self.pen_scratches_boxes.xywh.numpy()[i]
            conf = self.pen_scratches_boxes.conf[i]
            self.pen_scratches.append(BoundingBox(x, y, w, h, conf))

        # Hands bounding boxes
        self.hand_boxes = result.boxes
        number_hands_detected = len(self.hand_boxes)
        if number_hands_detected >= 1:
            self.hands = []
            for i in range(number_hands_detected):
                x, y, w, h = self.hand_boxes.xywh.numpy()[i]
                conf = self.hand_boxes.conf[i]
                self.hands.append(Hand(x, y, w, h, conf))

            self.hands.sort(key=lambda x: x.x)

            if number_hands_detected == 1:
                # Assumption that the only time that a single hand
                # is visible in the frame that hand is the right one.
                # The worker of the video always places the piece in the box
                # with his left hand. When he does so his left hand
                # doesn't appear in the frame at it is the only moment in 
                # the video where this occurs
                self.left_hand = None
                self.right_hand = self.hands[0]
                self.hands[0].location = "right"
            elif number_hands_detected == 2:
                self.left_hand, self.right_hand = self.hands
                self.hands[0].location = "left"
                self.hands[1].location = "right"
            else:
                raise MoreThan2HandsDetected(
                    "Frame with more than 2 hands detected",
                )

        else: # No hands detected
            raise NoHandsDetected("Frame without hands detected")
        
        if self.previous_frame is None:
            self.task_counter = OperationCounter(current_tasks_performed=[], operations=[])
        else:
            self.task_counter = deepcopy(self.previous_frame.task_counter)

        # Classify the frame task
        self.classification = self.task_classification()
        # Check the correctness of the task classification
        self.classification_status = self.check_classification()

        # If the current frame classification is wrong accordingly to the
        # operation task order then the classification reference for the 
        # next frame is most recent frame correctly classified
        if self.classification_status == ClassificationStatus.WRONG:
            self.classification_reference = self.previous_frame.classification
        else:
            self.classification_reference = self.classification
        
    @property
    def shape(self) -> tuple:
        """
        Dimensions of the video frame
        """
        return self.image.shape
    
    @staticmethod
    def calculate_interception(
        box_1: tuple[float, float, float, float],
        box_2: tuple[float, float, float, float,]
    ) -> float:
        """
        Calculate the interception over union of two bounding boxes
        """
        poly_1 = shapely.box(*box_1)
        poly_2 = shapely.box(*box_2)
        return poly_1.intersection(poly_2).area 
    
    def task_classification(self) -> TaskClassification:
        """
        Classifies each assembly moment
        """
        if self.left_hand is None:
            # Assumption: when the worker places the piece on the box
            # his left hand disapears from the frame
            self.classification = TaskClassification.Place
            return self.classification
        elif (self.right_hand.y - self.left_hand.y) > (self.right_hand.h / 2):
            # Assumption: when the worker picks up the pieces his left hand 
            # is significantly above his right hand
            self.classification = TaskClassification.Pick
            return self.classification
        elif Frame.calculate_interception(self.left_hand.xyxy, self.right_hand.xyxy) > 0.0:
            # Assumption: the only time that the hands bounding boxes
            # intercept is when the worker passes the prone through the piece holes
            self.classification = TaskClassification.PassProbe
            return self.classification
        elif len(self.pen_scratches_boxes):
            self.classification = TaskClassification.PenScratch
            return self.classification
        else:
            # If none of the four moments where detected one can use the previous frame
            # classification as the current frame classification (if this is not the first frame)
            if self.previous_frame is not None:  
                self.classification = self.previous_frame.classification
            # If this is the first frame and none of the four moments where detected
            else: 
                self.classification = TaskClassification.Unknown
            return self.classification
        
    def check_classification(self) -> ClassificationStatus:
        """
        Check if the current frame task classification matches the order of the assemblage operation tasks
        """
        if self.classification == TaskClassification.Unknown or self.previous_frame is None:
            return ClassificationStatus.UNKNOWN
        
        if self.classification == self.previous_frame.classification_reference:
            return self.previous_frame.classification_status
        
        # Task classification of the previous frame
        previous_frame_classification_index = ASSEMBLY_ORDER.index(
            self.previous_frame.classification_reference,
        )
        # Correct index of the assembly next task
        next_classification_index = (
            previous_frame_classification_index + 1 
            if previous_frame_classification_index < (len(ASSEMBLY_ORDER) - 1)
            else
            0
        )
        
        if self.classification == ASSEMBLY_ORDER[next_classification_index]:
            # Update the task counter if the current task matches the assemblage task order
            self.task_counter.update(self.classification, self.time)
            return ClassificationStatus.CORRECT
        else:
            return ClassificationStatus.WRONG
        
    def annotate_bounding_box(
        self, 
        image: Image, 
        bounding_box: BoundingBox,
        fill: str,
        text: str,
    ) -> Image:
        """
        Annotate a single bounding box with with the corresponding label
        """
        canva = image.copy()
        draw = ImageDraw.Draw(canva)
        draw.rectangle(bounding_box.xyxy, outline=fill)
        draw.text(
            bounding_box.xyxy[:2], 
            text,
            fill="blue",
            font=FONT,
        )

        return canva
        
    def annotate_hand_bounding_box(
        self, 
        image: Image, 
        bounding_box: BoundingBox,
    ) -> Image:
        """
        Annotate a single hand bounding box with with the corresponding location
        """
        canva = image.copy()
        draw = ImageDraw.Draw(canva)
        draw.rectangle(bounding_box.xyxy, outline="red")
        draw.text(
            bounding_box.xyxy[:2], 
            bounding_box.location,
            fill="blue",
            font=FONT,
        )

        return canva
    
    def annotate(self):
        """
        Annote every bounding box prediction with the corresponding label
        """
        canva = Image.fromarray(self.image.copy())
        for hand in self.hands:
            canva = self.annotate_hand_bounding_box(canva, hand)

        for pen_scratch in self.pen_scratches:
            canva = self.annotate_bounding_box(canva, pen_scratch, "purple", "")

        draw = ImageDraw.Draw(canva)
        draw.text(
            (20, 30), 
            f"Frame {str(self.frame_index + 1)}",
            fill="white",
            font=FONT,
        )
        draw.text(
            (20, 80), 
            f"Current Task: {self.classification.value} ({self.classification_status.name})",
            fill=self.classification_status.value,
            font=FONT,
        )
        draw.text(
            (20, 130),
            f"Number Completed Assemblies: {len(self.task_counter)}",
            fill="blue",
            font=FONT,
        )
        draw.text(
            (20, 180),
            "Assembly Duration (s)",
            fill="blue",
            font=FONT,
        )
        draw.text(
            (20, 220),
            f"Avg. Duration = {self.task_counter.avg_operation_duration():.4f}",
            fill="blue",
            font=FONT_SMALL,
        )
        draw.text(
            (20, 260),
            f"Min. Duration = {self.task_counter.min_operation_duration():.4f}",
            fill="blue",
            font=FONT_SMALL,
        )
        draw.text(
            (20, 300),
            f"Max. Duration = {self.task_counter.max_operation_duration():.4f}",
            fill="blue",
            font=FONT_SMALL,
        )
        draw.text(
            (20, 350),
            "Tasks Average Duration (s)",
            fill="yellow",
            font=FONT,
        )
        draw.text(
            (20, 390),
            f"Picking up piece: {self.task_counter.avg_task_duration(TaskClassification.Pick):.4f}",
            fill="yellow",
            font=FONT_SMALL,
        )
        draw.text(
            (20, 430),
            f"Passing Probe: {self.task_counter.avg_task_duration(TaskClassification.PassProbe):.4f}",
            fill="yellow",
            font=FONT_SMALL,
        )
        draw.text(
            (20, 470),
            f"Scratching Piece: {self.task_counter.avg_task_duration(TaskClassification.PenScratch):.4f}",
            fill="yellow",
            font=FONT_SMALL,
        )
        draw.text(
            (20, 510),
            f"Placing piece on the box: {self.task_counter.avg_task_duration(TaskClassification.Place):.4f}",
            fill="yellow",
            font=FONT_SMALL,
        )

        return canva
        

class AssemblyReader:

    """
    Assembly video processor which retrieves four distinct tasks:

    1. Person picks up the piece
    2. Probe passes through the piece
    3. Person makes a scratch on the piece
    4. Person places piece on the box

    Attributes:
        video_path: path for the source video to be processed
        weights_path: path for the YOLOv8 weights
        confidence_threshold: confidence threshold for filtering out 
                              model bounding box predictions 
        iou_threshold: interception-over-union threshold  used to determine 
                       whether two bounding boxes correspond to the same object
    """
    
    def __init__(
        self,
        video_path: str | Path,
        weights_path: str | Path,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.7,
        **kwargs,
    ):
        self.video_path = video_path

        self.model = YOLO(weights_path)

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        if "pen_scratch_detector" in kwargs:
            self.pen_scratch_detector = kwargs["pen_scratch_detector"]
        else:
            self.pen_scratch_detector = None

    def process_frame(
        self, 
        frame: np.ndarray, 
        frame_index: int,
        time_per_frame: float,
        previous_frame: Optional[Frame], 
    ) -> Frame:
        """
        Run model on the input frame retrieving a prediction
        abstraction responsible for classifying each frame task

        Parameters:
            frame: source video current frame array
            frame_index: index of the current frame array
            time_per_frame: duration between two consecutive frames
            previous_frame: previous frame processed
            
        Returns:
            an assembly frame prediction abstraction
        """
        result = self.model(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
        )[0]

        return Frame(
            result, 
            frame_index=frame_index, 
            time_per_frame=time_per_frame,
            previous_frame=previous_frame, 
            pen_scratch_detector=self.pen_scratch_detector,
        )
    
    def process_video(self, max_frames: Optional[int] = None) -> list[Frame]:
        """
        Detect assembly tasks for every frame in the source video

        Parameters:
             max_frames: maximum number of frames to process
        """
        video_info = sv.VideoInfo.from_video_path(video_path=self.video_path)
        time_per_frame = 1 / video_info.fps
        frame_generator = sv.get_video_frames_generator(source_path=self.video_path)

        processed_frame = None
        processed_frames = []
        print("Running ...")
        for i, frame in enumerate(frame_generator):
            
            if max_frames is not None and (i + 1) > max_frames:
                break

            try:
                processed_frame = self.process_frame(
                    frame, 
                    frame_index=i,   
                    time_per_frame=time_per_frame,
                    previous_frame=processed_frame,   
                )
            except MoreThan2HandsDetected as e:
                print(e)
                continue

            processed_frames.append(processed_frame)
        
        print("Done.")
        return processed_frames

