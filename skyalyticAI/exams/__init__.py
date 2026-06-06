"""考试题型模块：字符预测、阅读理解、多步推理。"""

from skyalyticAI.exams.exam_suite import ExamSuite, ExamType
from skyalyticAI.exams.char_prediction_exam import CharPredictionExam
from skyalyticAI.exams.reading_comprehension_exam import ReadingComprehensionExam
from skyalyticAI.exams.multi_step_reasoning_exam import MultiStepReasoningExam

__all__ = [
    "ExamSuite",
    "ExamType",
    "CharPredictionExam",
    "ReadingComprehensionExam",
    "MultiStepReasoningExam",
]
