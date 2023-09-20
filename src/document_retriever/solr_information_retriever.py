import glob
import json
import os
import time
from ..db_opts.qdrant_db import QdrantDocRetriever
from ..db_opts.solr_db import SolrDocRetriever
from ..base.base import BaseClass
from ..base.feedback_loop import JsonFileDrProcessedFilesOpts
from ..document_processor.generate_docs_for_vector_db import DocsForDb


class DocRetriver(BaseClass):
    '''
    document retriever.
    '''
    def __init__(self) -> None:
        super().__init__()
        document_path = self.doc_retrieval_config['document_path']

        self.nlp_preprocessing = self.doc_retrieval_config['nlp_preprocessing']
        self.nlp_preprocessing_topics = self.nlp_preprocessing['topics']
        self.nlp_preprocessing_summary = self.nlp_preprocessing['summary']

        self.vector_db_path = self.doc_retrieval_config['vector_db_path']

        self.document_upload_structure = self.doc_retrieval_config['document_upload_structure']

        self.use_page_content_similarity = self.document_upload_structure['use_page_content_similarity']
        self.use_page_content_similarity_useit = self.use_page_content_similarity['use_it']
        self.use_page_content_similarity_cname = self.use_page_content_similarity['collection_name']

        self.use_topic_similarity = self.document_upload_structure['use_topic_similarity']
        self.use_topic_similarity_useit = self.use_topic_similarity['use_it']
        self.use_topic_similarity_cname = self.use_topic_similarity['collection_name']

        self.use_both = self.document_upload_structure['use_both']
        self.use_both_useit = self.use_both['use_it']
        self.use_both_content_cname = self.use_both['content_collection_name']
        self.use_both_topic_cname = self.use_both['topic_collection_name']

        self.duplicate_question_finder = self.doc_retrieval_config['duplicate_question_finder']
        self.duplicate_question_finder_useit = self.duplicate_question_finder['use_it']
        self.duplicate_question_finder_content_cname = self.duplicate_question_finder['collection_name']

        self.use_vector_embedding = self.doc_retrieval_config['use_vector_embedding']
        self.use_vector_embedding_model = None
        self.max_seq_length = 150
        self.use_lower_embedding_size = False
        for each_model_config_key in self.use_vector_embedding:
            if self.use_vector_embedding[each_model_config_key]['use_it']:
                if self.use_vector_embedding[each_model_config_key]['use_local_model']['use_it']:
                    self.use_vector_embedding_model = self.use_vector_embedding[each_model_config_key]['use_local_model']['model']
                else:
                    self.use_vector_embedding_model = self.use_vector_embedding[each_model_config_key]['model']
                self.max_seq_length = self.use_vector_embedding[each_model_config_key]['max_seq_length']
                self.use_lower_embedding_size = self.use_vector_embedding[each_model_config_key].get('use_lower_embedding_size',False)

        self.files = DocRetriver.get_files(document_path)

        self.processed_files_opts = JsonFileDrProcessedFilesOpts()
        self.processed_files_opts.create_file()
        self.processed_files_opts.create_error_file()
        self.processed_files_data = self.processed_files_opts.read_data()
        self.processed_files_error_data = self.processed_files_opts.read_error_data()
        self.processed_files = self.processed_files_data['data']
        self.processed_error_files = self.processed_files_error_data['data']

        self.docs_for_db = DocsForDb()
        
        self.solr_obj = SolrDocRetriever(solr_address=self.doc_retrieval_config['use_db']['solr_config']['url'],
                                            encoder_model=self.use_vector_embedding_model)
        
        if self.duplicate_question_finder_useit:
            self.qdrant_qa_obj = QdrantDocRetriever(self.vector_db_path,
                                                 self.duplicate_question_finder_content_cname,
                                                 self.use_vector_embedding_model,
                                                 self.use_lower_embedding_size)

    @staticmethod
    def get_files(file_path):
        files = glob.glob(file_path)
        return files
    
    def process_data(self,files=None):
        if files is None:
            files = self.files
        
        start_time_total = time.time()
        for each_file_path in files:
            if not self.processed_files.get(each_file_path,None):
                try:
                    self.logger.info(f"processing file: {each_file_path}")
                    start_time = time.time()
                    # generate documents
                    document_list,keyword_output = self.docs_for_db.get_structured_docs_for_db(each_file_path,
                                                                                               self.max_seq_length,
                                                                                                kwargs = {'topics':self.nlp_preprocessing_topics,
                                                                                                                    'data_summary':self.nlp_preprocessing_summary,
                                                                                                                    'openai_url':self.openai_url,
                                                                                                                    'openai_token':self.openai_token,
                                                                                                                    'openai_model':self.openai_model,
                                                                                                                    'openai_max_token':self.openai_max_token,
                                                                                                                    'openai_temperature':self.openai_temperature}
                                                                                                )
                    self.logger.info(f"To process one pdf with record size({len(document_list)}) and topics size ({len(keyword_output)}). time took: {time.time()-start_time}")
                    start_time = time.time()
                    # upload documents
                    self.solr_obj.upload_data_records(documents=document_list)
                    self.logger.info(f"To upload of records size({len(document_list)}). time took: {time.time()-start_time}")

                    # upload topics
                    if self.use_topic_similarity_useit or self.use_both_useit:
                        start_time = time.time()
                        self.qdrant_topics_obj.upload_data_records(key='keywords_text',documents=keyword_output)
                        self.logger.info(f"To upload of topics size({len(keyword_output)}). time took: {time.time()-start_time}")
                    
                    # add to processed files
                    self.processed_files[each_file_path] = 1
                except Exception as err:
                    self.logger.error(err)
                    self.processed_error_files[each_file_path] = str(err)
        
        # add to processed files
        self.processed_files_opts.add_data(self.processed_files)
        self.processed_files_opts.add_error_data(self.processed_error_files)

        self.logger.info(f"total time took: {time.time()-start_time_total}")
    
    def upload_data_records(self,key="page_content",documents=[]):
        # upload documents
        self.solr_obj.upload_data_records(key=key,documents=documents)
    
    def get_similar_docs(self,text,limit=5):
        '''
        call qdrant-db api, to get top match records.
        use global config for query.
        '''
        match_found = []
        topic_match_found = []
        if self.use_page_content_similarity_useit:
            match_found = self.solr_obj.get_records(text,limit)
        elif self.use_topic_similarity_useit:
            topic_match_found = []#self.qdrant_topics_obj.get_records(text,limit)
        elif self.use_both_useit:
            topic_match_found = []#self.qdrant_topics_obj.get_records(text,limit)
            match_found = self.solr_obj.get_records(text,limit)

        return (match_found,topic_match_found)
    
    def get_similar_docs_old(self,text,limit=5):
        match_found = self.solr_obj.get_records(text,limit)

        return match_found
    
    def get_similar_docs_with_filter(self,text,filter='',limit=5):
        match_found = self.solr_obj.get_records_with_filter(text,filter,limit)

        return match_found
    
    def get_similar_topic_docs(self,text,limit=5):
        '''
        match topic vectors, then return matched records.
        '''
        if self.use_topic_similarity_useit or self.use_both_useit:
            match_found = self.qdrant_topics_obj.get_records(text,limit)
        else:
            raise Exception("config setup hasn't been done correctly.")

        return match_found
    
    def get_similar_topic_docs_with_filter(self,text,filter=[],limit=5):
        '''
        match topic vectors and other filter conditions, then return matched records.
        '''
        if self.use_topic_similarity_useit or self.use_both_useit:
            match_found = self.qdrant_topics_obj.get_records_with_multi_filter_direct(text,filter,limit)
        else:
            raise Exception("config setup hasn't been done correctly.")

        return match_found
    
    def get_similar_doc_contents(self,text,limit=5):
        '''
        match text/content vectors and return matched records.
        '''
        (match_found,topic_match_found) = self.get_similar_docs(text,limit)
        
        return (match_found,topic_match_found)
        doc_contents = ""
        self.logger.info(f"top {limit} match docs are...")
        for hit in hits:
            self.logger.info("payload: ",hit.payload, "score: ", hit.score)
            doc_contents += hit.payload['page_content']
        
        return doc_contents
    
    def get_similar_doc_windowed_contents(self,text,limit=5,docs_window=5):
        '''
        match text/content vectors and get sorrounding neighbors using window-size defined.
        then return matched records.
        '''
        doc_contents = self.solr_obj.get_records_window(text,limit,docs_window)

        return doc_contents
    
    def get_similar_docs_qa(self,text,limit=5):
        '''
        match text vectors and return matched records.
        '''
        match_found = self.qdrant_qa_obj.get_records_dict(text,limit)

        return match_found



if __name__ == '__main__':
    docprocessor = DocRetriver()
    docprocessor.process_data()

    matched_topic_records = docprocessor.get_similar_topic_docs("give me former employer details from CONTINGENT WORKER CONFIDENTIALITY AGREEMENT ?")
    print(matched_topic_records)
    print("\n===========\n")
    (match_found,topic_match_found) = docprocessor.get_similar_docs("give me former employer details from CONTINGENT WORKER CONFIDENTIALITY AGREEMENT ?")
    print(match_found)