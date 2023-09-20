import glob
import sys
import os
import json
import io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from ..db_opts.qdrant_db import QdrantDocRetriever
from ..base.base import BaseClass
from ..base.feedback_loop import SqliteDbOpts,JsonFileDrProcessedSqliteDataOpts


class FeedbackQnaProcessor(BaseClass):
    '''
    question & answer processor.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.feedback_loop_path_qa_sqlite = self.doc_retrieval_config['feedback_loop_path_qa_sqlite']
        self.feedback_loop_path_qa_sqlite_table = self.doc_retrieval_config['feedback_loop_path_qa_sqlite_table']

        self.vector_db_path = self.doc_retrieval_config['vector_db_path']
        self.duplicate_question_finder = self.doc_retrieval_config['duplicate_question_finder']
        self.duplicate_question_finder_useit = self.duplicate_question_finder['use_it']
        self.duplicate_question_finder_content_cname = self.duplicate_question_finder['collection_name']
        self.use_vector_embedding = self.doc_retrieval_config['use_vector_embedding']
        self.use_vector_embedding_model = None
        self.max_seq_length = 150
        self.use_lower_embedding_size = False
        for each_model_config_key in self.use_vector_embedding:
            if self.use_vector_embedding[each_model_config_key]['use_it']:
                self.use_vector_embedding_model = self.use_vector_embedding[each_model_config_key]['model']
                self.max_seq_length = self.use_vector_embedding[each_model_config_key]['max_seq_length']
                self.use_lower_embedding_size = self.use_vector_embedding[each_model_config_key].get('use_lower_embedding_size',False)

        self.processed_qna_data_opts = JsonFileDrProcessedSqliteDataOpts()
        self.processed_qna_data_opts.create_file()
        self.processed_qna_data = self.processed_qna_data_opts.read_data()

        if self.duplicate_question_finder_useit:
            self.qdrant_obj = QdrantDocRetriever(self.vector_db_path,
                                                 self.duplicate_question_finder_content_cname,
                                                 self.use_vector_embedding_model,
                                                 self.use_lower_embedding_size)
    
    def process_data(self,document_list=None):
        try:
            # read json, get max_id
            max_id_processed = self.processed_qna_data.get('max_index',0)
            
            if document_list is None:
                # get data from sqlite
                feedback_loop_obj = SqliteDbOpts(self.feedback_loop_path_qa_sqlite,self.feedback_loop_path_qa_sqlite_table)
            
                document_list = feedback_loop_obj.get_records(where_clause=f"Id > {max_id_processed} AND Flag != 0")

            if len(document_list) == 0:
                self.logger.info(f"no qa records to be processed.")
                return None

            # upload documents
            self.qdrant_obj.upload_data_records(key="question",documents=document_list)

            # add to processed files
            max_id_processed = max([i['id'] for i in document_list])
            self.processed_qna_data['max_index'] = max_id_processed
        except Exception as err:
            #print(err)
            self.logger.error(err)
        
        # add to processed files
        self.processed_qna_data_opts.add_data(self.processed_qna_data)
    
    def get_similar_docs(self,text,limit=5):
        match_found = self.qdrant_obj.get_records_dict(text,limit)

        return match_found


if __name__ == '__main__':
    docprocessor = FeedbackQnaProcessor()
    docprocessor.process_data()

    matched_records = docprocessor.get_similar_docs("what is the enrollment process for microsoft's solution for password management ?")
    print(matched_records)