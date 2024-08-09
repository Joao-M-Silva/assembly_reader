"""
Utils to count and analyse every operation on a source video
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class TaskClassification(Enum):

    """
    Unique assembly tasks 
    """

    Pick = "Picking up piece"
    PassProbe = "Passing probe"
    PenScratch = "Scratching Piece"
    Place = "Placing piece on the box" 
    Unknown = "Unknown"


ASSEMBLY_ORDER = (
    TaskClassification.Pick,
    TaskClassification.PassProbe,
    TaskClassification.PenScratch,
    TaskClassification.Place,
)


@dataclass
class Task:

    """
    Representation of a single task

    Attributes:
        classification: unique assembly task
        time: current time in the referential of the source video.
              time = (1 / fps) * frame_number
        duration: time interval between the beggining and end of the task 
    """

    classification: TaskClassification
    time: float
    duration: Optional[float]


class Operation:

    """
    Representation of a single operation.
    Each operation cicle consists on the given sequence:
    Pick -> Pass Probe -> Pen Scratch -> Place 

    Attributes:
        tasks_performed: tasks performed in the operation
    """

    def __init__(self, tasks_performed : list[Task]):
        # Assumption: each operation must have all the tasks concluded
        assert len(tasks_performed) == len(ASSEMBLY_ORDER) 
        # The flux of tasks is considered to be consistent, in that way
        # a task only ends when the next task initiates. As so, in order to
        # know the duration of a given task the next task needs to start
        assert tasks_performed[-1].duration is None

        self.tasks_performed = tasks_performed

    @property
    def duration(self) -> float:
        """
        Duration of a full operation cicle
        """
        return np.sum(
            task.duration
            for task in self.tasks_performed[:(len(ASSEMBLY_ORDER)-1)]
        )
    
    @property
    def tasks_duration(self) -> dict[TaskClassification, Optional[float]]:
        """
        Duration of each operation task
        Note: due to the task continuity assumption one can't know the 
        duration of the last task before the next operation first task starts
        """
        return {
            task.classification: task.duration
            for task in self.tasks_performed
        }

    def __getitem__(self, i) -> Task:
        """
        Return the ith task
        """
        return self.tasks_performed[i]
    
    def __len__(self) -> int:
        """
        Return the number of tasks on a full operation cycle
        """
        return len(self.tasks_performed)
        

@dataclass
class OperationCounter:

    """
    Counts and stores the durations of every task in an operation

    Attributes:
        current_tasks_performed: tasks performed in the current operation cycle
        operations: full operation cycles already performed 
    """

    def __init__(self, current_tasks_performed: list[Task], operations: list[Operation]):
        self.current_tasks_performed = current_tasks_performed
        self.operations = operations 

    def update(self, current_frame_classification: TaskClassification, time: int):
        """
        Update the tasks performed in the current operation

        Note: when a new tasks is added to the operation cycle its duration its
        unknown since it is considered task continuity in this project. That is, 
        a current task duration corresponds to the time interval between the beggining
        of the task and start of the next task in the sequence of the assembly tasks

        Attributes:
            current_frame_classification: current frame task classification
            time: time of the frame considering the as referential origin
                  the beggining of the source video.
                  time = (1 / fps) * frame_number
        """
        if self.current_tasks_performed == []: 
            # Another operation has not started yet after the conclusion of the 
            # previous operation
            if current_frame_classification == ASSEMBLY_ORDER[0]:
                # The current task classification correspondes to the first
                # task in the operation cycle
                if self.operations:
                    # Update the previous operation last task duration based
                    # on the time of the current frame
                    time_previous_task = self.operations[-1][-1].time
                    self.operations[-1][-1].duration = time - time_previous_task
                
                self.current_tasks_performed.append(
                    Task(
                        classification=current_frame_classification,
                        time=time,
                        duration=None,
                    )
                )
                
        else:
            # Only add unique tasks to the operation cycle representation
            if current_frame_classification not in [task.classification for task in self.current_tasks_performed]:
                
                # Update the duration of the previous performed task
                time_previous_task = self.current_tasks_performed[-1].time
                self.current_tasks_performed[-1].duration = time - time_previous_task

                self.current_tasks_performed.append(
                    Task(
                        classification=current_frame_classification,
                        time=time,
                        duration=None,
                    )
                )
                
                # An operation is concluded when the last task in the operation cycle starts
                if len(self.current_tasks_performed) == len(ASSEMBLY_ORDER):
    
                    self.operations.append(Operation(self.current_tasks_performed))
                    # Reset the current operation
                    self.current_tasks_performed = []

    def __getitem__(self, i: int) -> Operation:
        """
        Retrieve the ith operation
        """
        return self.operations[i]
    
    def __len__(self) -> int:
        """
        Retrieve the number of operations tracked
        """
        return len(self.operations)
    
    def avg_operation_duration(self) -> float:
        """
        Retrieves the average operation duration
        """
        if self.operations:
            return np.mean(
                [
                    operation.duration 
                    for operation in self.operations 
                    if operation.duration is not None
                ]
            )
        else:
            return 0.0
        
    def avg_task_duration(self, task: TaskClassification) -> float:
        """
        Retrieves the average duration of a given task 

        Parameters:
            task: type of task to analyse
        """
        if self.operations:
            return np.mean(
                [
                    operation.tasks_duration[task] 
                    for operation in self.operations
                    if operation.tasks_duration[task] is not None
                ]
            )
        else:
            return 0.0
        
    def min_operation_duration(self) -> float:
        """
        Retrieves the fastest operation duration
        """
        if self.operations:
            return np.min(
                [
                    operation.duration 
                    for operation in self.operations
                    if operation.duration is not None
                ]
            )
        else:
            return 0.0
        
    def min_task_duration(self, task: TaskClassification) -> float:
        """
        Retrieves the fastest duration of a given task 

        Parameters:
            task: type of task to analyse
        """
        if self.operations:
            return np.min(
                [
                    operation.tasks_duration[task] 
                    for operation in self.operations
                    if operation.tasks_duration[task] is not None
                ]
            )
        else:
            return 0.0
    
    def max_operation_duration(self) -> float:
        """
        Retrieves the slowest operation duration
        """
        if self.operations:
            return np.max(
                [
                    operation.duration 
                    for operation in self.operations
                    if operation.duration is not None
                ]
            )
        else:
            return 0.0
        
    def max_task_duration(self, task: TaskClassification) -> float:
        """
        Retrieves the slowest duration of a given task 

        Parameters:
            task: type of task to analyse
        """
        if self.operations:
            return np.max(
                [
                    operation.tasks_duration[task] 
                    for operation in self.operations
                    if operation.tasks_duration[task] is not None
                ]
            )
        else:
            return 0.0