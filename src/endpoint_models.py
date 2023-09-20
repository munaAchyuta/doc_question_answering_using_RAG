from typing import Dict, List, Optional, Union
from pydantic import BaseModel


class DocItem(BaseModel):
    input_text: str
    other: Optional[str] = ""

class DocFeedbackItem(BaseModel):
    input_text: str
    answer: Optional[str] = ""
    other: Optional[List[Dict]] = []
    flag: int = 0

class QuestionAnswerItem(BaseModel):
    question: str
    context: str

class TableQuestionAnswerItem(BaseModel):
    question: str
    table: List[Dict]

class ProcessDocItem(BaseModel):
    file_path: List[str] = None