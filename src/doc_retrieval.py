"""Main module."""
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Optional, Union
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.exceptions import HTTPException
import aiofiles
import pandas as pd

from .base.base import BaseClass
from .document_retriever.response_generator import ResponseGenerator
from .endpoint_models import (DocItem,
                              DocFeedbackItem,
                              QuestionAnswerItem,
                              TableQuestionAnswerItem,
                              ProcessDocItem)
from .document_processor.processor import DocProcessor
from .document_processor.doc_retrieval_feedback_processor import FeedbackQnaProcessor

baseclass_obj = BaseClass()
logger = baseclass_obj.logger

response_generator = ResponseGenerator()
doc_processor = DocProcessor().doc_proc
doc_qa_processor = FeedbackQnaProcessor()

app = FastAPI()


@app.get("/")
def read_root():
    '''
    ROOT.
    '''
    logger.info(f"root API.")

    return {"Hello": "you made it."}

@app.put("/upload_file/")
async def upload_file(in_file: UploadFile=File(...)):
    out_file_path = baseclass_obj.doc_retrieval_config['document_path']
    out_file_path = '/'.join(out_file_path.split('/')[:-1]) + f'/{in_file.filename}'

    async with aiofiles.open(out_file_path, 'wb') as out_file:
        content = await in_file.read()
        await out_file.write(content)

    return {"status": "success","file_path":out_file_path}

@app.put("/process_document/")
def process_document(item:ProcessDocItem):
    '''
    get_answer_given_question_table.
    '''
    logger.info(f"process_document API.")

    _ = doc_processor.process_data(item.file_path)

    return {'status':'success'}

@app.get("/get_doc_names/")
def get_doc_names():
    '''
    get get_doc_names data from path
    '''
    logger.info(f"get_doc_names API.")
    
    files = ResponseGenerator().doc_retriever.files

    return {"status":True,"files":files}

@app.post("/retrive_document/")
def retrive_document(item: DocItem):
    '''
    input: text, other
    output: retrieve document
    '''
    logger.info(f"retrive_document API input: {item.__dict__}")
    
    try:
        # use duplicate question finder
        if baseclass_obj.doc_retrieval_config['duplicate_question_finder']['use_it']:
            logger.info("using duplicate question finder.")
            duplicates = response_generator.get_top_duplicate_records(item.input_text)
            for each_rec in duplicates:
                if each_rec['score'] > baseclass_obj.doc_retrieval_config['duplicate_question_finder']['match_threshold']:
                    logger.info(f"duplicate question found with score: {each_rec['score']}")
                    logger.info(f"duplicate question is: {each_rec['payload']['question']}")
                    response = each_rec.copy()
                    response['answer'] = each_rec['payload']['answer']
                    response['duplicate'] = each_rec['payload']['question']
                    return response
            else:
                logger.info(f"duplicate question not found.")
        
        # retrieve document
        retrieved_data = response_generator.get_response(item.input_text)
        

        logger.info(f"retrive_document API summarized_data: {retrieved_data}")

        response = {'output': retrieved_data}
    except Exception as err:
        logger.error(err)
        raise HTTPException(409, f"Error: {str(err)}")

    return response

@app.put("/add_feedback_loop_qa_data/")
def add_feedback_loop_qa_data(item:DocFeedbackItem):
    '''
    add_feedback_loop_qa_data for Document Retriever.
    '''
    logger.info(f"add_feedback_loop_qa_data API.")

    _ = response_generator.feedback_loop_obj_qa.add_feedback_loop_data({'api':'document_retriever',
                                            'question':item.input_text,
                                            'answer':item.answer,
                                            'other':item.other,
                                            'flag':item.flag}
                                            )
    
    _ = doc_qa_processor.process_data()
    
    return {"status":True}

@app.get("/get_feedback_loop_data/")
def read_feedback_loop_data():
    '''
    get feedback-loop data from DB
    '''
    logger.info(f"get_feedback_loop_data API.")

    data = response_generator.feedback_loop_obj.get_records()

    return data

@app.get("/get_feedback_loop_qa_data/")
def read_feedback_loop_qa_data():
    '''
    get read_feedback_loop_qa_data data from DB
    '''
    logger.info(f"read_feedback_loop_qa_data API.")
    
    data = response_generator.feedback_loop_obj_qa.get_records()

    return data

@app.post("/get_answer_given_question_context/")
def get_answer_given_question_context(item:QuestionAnswerItem):
    '''
    get_answer_given_question_context.
    '''
    logger.info(f"get_answer_given_question_context API.")

    response = response_generator.extract_answer_from_question_given_context(item.question,item.context)

    return response

@app.post("/get_answer_given_question_table/")
def get_answer_given_question_table(item:TableQuestionAnswerItem):
    '''
    get_answer_given_question_table.
    '''
    logger.info(f"get_answer_given_question_table API.")

    table = pd.DataFrame(item.table)
    table = table.astype(str)
    response = response_generator.extract_answer_from_question_given_tablecontext(item.question,table)

    return response
