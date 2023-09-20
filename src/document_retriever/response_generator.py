from nltk.tokenize import sent_tokenize, word_tokenize

from ..base.base import BaseClass
from .information_retriever import DocRetriver
from .summarizer import Summarizer
from ..base.feedback_loop import SqliteDbOpts
from .question_answer import QuestionAnsweringModel


class ResponseGenerator(BaseClass):
    '''
    this one acts as middle layer between rest API and DB operations.
    '''
    def __init__(self) -> None:
        super().__init__()
        # create sqlite db for logging top-N match records
        self.feedback_loop_path_dr_sqlite = self.doc_retrieval_config['feedback_loop_path_dr_sqlite']
        self.feedback_loop_path_dr_sqlite_table = self.doc_retrieval_config['feedback_loop_path_dr_sqlite_table']
        self.feedback_loop_obj = SqliteDbOpts(feedback_loop_path_sqlite=self.feedback_loop_path_dr_sqlite,
                                              feedback_loop_path_sqlite_table=self.feedback_loop_path_dr_sqlite_table)
        
        # create sqlite db for logging question-answer feedback
        self.feedback_loop_path_qa_sqlite = self.doc_retrieval_config['feedback_loop_path_qa_sqlite']
        self.feedback_loop_path_qa_sqlite_table = self.doc_retrieval_config['feedback_loop_path_qa_sqlite_table']
        self.feedback_loop_obj_qa = SqliteDbOpts(feedback_loop_path_sqlite=self.feedback_loop_path_qa_sqlite,
                                              feedback_loop_path_sqlite_table=self.feedback_loop_path_qa_sqlite_table)
        
        self.document_retrieval_structure = self.doc_retrieval_config['document_retrieval_structure']

        self.return_similar_docs = self.document_retrieval_structure['return_similar_docs']
        self.return_similar_docs_useit = self.return_similar_docs['use_it']
        self.retrieval_docs_limit = self.doc_retrieval_config['retrieval_docs_limit']
        self.return_similar_docs_header_to_text = self.return_similar_docs['use_of_retrieval_order']['header_to_text']
        self.return_similar_docs_header_topic_to_text = self.return_similar_docs['use_of_retrieval_order']['header_topic_to_text']

        self.return_summary = self.document_retrieval_structure['return_summary']
        self.return_summary_useit = self.return_summary['use_it']
        self.return_summary_header_to_text = self.return_summary['use_of_retrieval_order']['header_to_text']
        self.return_summary_header_topic_to_text = self.return_summary['use_of_retrieval_order']['header_topic_to_text']

        self.return_docs_using_question_answer = self.document_retrieval_structure['return_docs_using_question_answer']
        self.return_docs_using_question_answer_useit = self.return_docs_using_question_answer['use_it']
        self.return_docs_using_question_answer_header_to_text_then_qa = self.return_docs_using_question_answer['use_of_retrieval_order']['header_to_text_then_qa']
        self.return_docs_using_question_answer_header_topic_to_text_then_qa = self.return_docs_using_question_answer['use_of_retrieval_order']['header_topic_to_text_then_qa']

        self.use_reranking = self.document_retrieval_structure['use_reranking']

        self.question_answer_config = self.document_retrieval_structure['return_docs_using_question_answer']['question_answer_config']
        self.retrieval_docs_window = self.doc_retrieval_config['retrieval_docs_window']

        self.doc_retriever = DocRetriver().doc_retriever

        self.summarizer = Summarizer()
        self.question_answer_model = QuestionAnsweringModel(self.question_answer_config['qa_model'],
                                                            self.question_answer_config['tqa_model'],
                                                            self.question_answer_config['similarity_model'])
        
    
    def rerank_docs(self,retrieved_data,input_text):
        sorted_docs = []
        for ind,data in enumerate(retrieved_data):
            if data.get('page_data',None):
                page_data = data['page_data']
            else:
                page_data = data['payload']['page_content']
            
            sentences = sent_tokenize(page_data)
            sentences = [i.strip().replace('\n','').replace('_','').lower() for i in sentences]
            sentences = "".join(sentences)

            sim_score = self.question_answer_model.similarity_score(sentences,input_text)
            sorted_docs.append((data,sim_score))
        sorted_docs = sorted(sorted_docs,key=lambda x: x[1],reverse=True)
        
        return [i[0] for i in sorted_docs]
    
    def generate_summary_response_using_window(self,input_text,limit=5):
        '''
        1. get top match records but with windowed neighbors output.
        2. then use summarizer model to summerize.
        '''
        retrieved_data = self.doc_retriever.get_similar_doc_windowed_contents(input_text,limit,self.retrieval_docs_window)

        if self.use_reranking:
            retrieved_data = self.rerank_docs(retrieved_data,input_text)

        for ind,small_chunk in enumerate(retrieved_data):
            # summarize it
            summarized_small_chunk_output = self.summarizer.summarize(small_chunk['text'],input_text)
            retrieved_data[ind]['answer'] = summarized_small_chunk_output
        
        return retrieved_data
    
    def generate_summary_response(self,input_text,limit=5):
        '''
        1. get top match records but with windowed neighbors output.
        2. then use summarizer model to summerize.
        '''
        retrieved_data = self.top_similar_docs(input_text,limit)

        if self.use_reranking:
            retrieved_data = self.rerank_docs(retrieved_data,input_text)

        for ind,data in enumerate(retrieved_data):
            page_data = data['page_data']
            sentences = sent_tokenize(page_data)
            sentences = [i.strip().replace('\n','').replace('_','').lower() for i in sentences]

            
            sent_match_score = []
            for each_sent in sentences:
                sim_score = self.question_answer_model.similarity_score(each_sent,input_text)
                sent_match_score.append((each_sent,sim_score))
            sent_match_score = sorted(sent_match_score,key=lambda x: x[1],reverse=True)
            sent_match_score = [i[0] for i in sent_match_score]
            sent_match_score = '.'.join(sent_match_score)
            page_data = sent_match_score[:1600]

            # summarize it
            summarized_small_chunk_output = self.summarizer.summarize(page_data,input_text)
            retrieved_data[ind]['answer'] = summarized_small_chunk_output
        
        return retrieved_data
    
    def top_similar_docs_window(self,input_text,limit=5):
        '''
        1. get top match records but with windowed neighbors output.
        '''
        retrieved_data = self.doc_retriever.get_similar_doc_windowed_contents(input_text,limit,self.retrieval_docs_window)
        
        if self.use_reranking:
            retrieved_data = self.rerank_docs(retrieved_data,input_text)

        for i,_ in enumerate(retrieved_data):
            retrieved_data[i]['answer'] = retrieved_data[i]['text']
            retrieved_data[i].pop('text')

        return retrieved_data
    
    def top_similar_docs(self,input_text,limit=5):
        '''
        1. get top match records.
        '''
        (match_found,topic_match_found) = self.doc_retriever.get_similar_doc_contents(input_text,limit)
        match_found = [i.dict() for i in match_found]
        topic_match_found = [i.dict() for i in topic_match_found]

        if len(topic_match_found) == 0:
            return match_found
        
        for i,_ in enumerate(match_found):
            # get page-data using file-name and page-number.
            filter_dict = [{'file_name':match_found[i]['payload']['file_name'],
                            'page_number':match_found[i]['payload']['page_number']}]
            tp_match_found = self.doc_retriever.get_similar_topic_docs_with_filter(input_text,filter_dict,limit)
            match_found[i]['page_data'] = tp_match_found[0].payload['raw_data']
        
        if self.use_reranking:
            match_found = self.rerank_docs(match_found,input_text)
        
        return match_found
    
    def top_similar_docs_qa(self,input_text,limit=5):
        '''
        1. get top match records.
        2. use qa-model and tqa-model to extract factual/descriptive answer.
        '''
        (match_found,topic_match_found) = self.doc_retriever.get_similar_doc_contents(input_text,limit)
        match_found = [i.dict() for i in match_found]
        topic_match_found = [i.dict() for i in topic_match_found]

        # get unique file with page
        unq_match_found = {}
        match_found_unq = []
        for j,_ in enumerate(match_found):
            tt = f"{match_found[j]['payload']['file_name']}_{str(match_found[j]['payload']['page_number'])}"
            if not unq_match_found.get(tt,None):
                match_found_unq.append(match_found[j])
                unq_match_found[tt] = 1
        match_found = match_found_unq.copy()
        
        for i,_ in enumerate(match_found):
            # get page-data using file-name and page-number.
            filter_dict = [{'file_name':match_found[i]['payload']['file_name'],
                            'page_number':match_found[i]['payload']['page_number']}]
            tp_match_found = self.doc_retriever.get_similar_topic_docs_with_filter(input_text,filter_dict,limit)
            match_found[i]['page_data'] = tp_match_found[0].payload['raw_data']

            # check for table present
            if match_found[i]['payload']['table_present'] and match_found[i]['payload']['is_table']:
                # call table question & answer model
                table = tp_match_found[0].payload['table']
                table_answer_list = []
                for each_table in table:
                    table_answer = self.question_answer_model.get_answer_given_question_table(input_text,each_table)
                    table_answer_list.append(table_answer)
                match_found[i]['answer'] = table_answer_list
            else:
                # call question & answer model
                context = tp_match_found[0].payload['raw_data']
                context_answer = self.question_answer_model.get_answer_given_question_context(input_text,context)
                match_found[i]['answer'] = context_answer
        
        if self.use_reranking:
            match_found = self.rerank_docs(match_found,input_text)

        return match_found
    
    def top_similar_topics(self,input_text,limit=5):
        '''
        given input: text
        1. search on topic vectors on qdrant db, get top match records.
        '''
        (match_found,topic_match_found) = self.doc_retriever.get_similar_doc_contents(input_text,limit)
        match_found = [i.dict() for i in match_found]
        topic_match_found = [i.dict() for i in topic_match_found]

        if self.use_reranking:
            topic_match_found = self.rerank_docs(topic_match_found,input_text)

        #for i,_ in enumerate(topic_match_found):
        #    topic_match_found[i]['answer'] = topic_match_found[i]['payload']['keywords_text']
        
        for i,_ in enumerate(topic_match_found):
            # get page-data using file-name and page-number.
            filter_dict = [{'file_name':topic_match_found[i]['payload']['file_name'],
                            'page_number':topic_match_found[i]['payload']['page_number']}]
            tp_match_found = self.doc_retriever.get_similar_topic_docs_with_filter(input_text,filter_dict,limit)
            topic_match_found[i]['page_data'] = tp_match_found[0].payload['raw_data']
        
        return topic_match_found
    
    def hierarchy_match(self,text,limit=5):
        '''
        1. get top files
        2. then search over those file, find top pages
        3. then search over those page, find top contents
        4. return either page text or summary or window-contents
        '''
        pass
    
    def get_response(self,text,limit=5):
        '''
        this function returns response based on input config.

        NOTE: in the logic, under each if condition, same function used. but,
            based on your project data best suite, please modify existing functions or add functions.
        '''
        limit = self.retrieval_docs_limit
        data = []
        text_bkp = text[:]
        text = text.lower()

        if self.return_similar_docs_useit:
            if self.return_similar_docs_header_to_text:
                data = self.top_similar_topics(text,limit)
            elif self.return_similar_docs_header_topic_to_text:
                data = self.top_similar_topics(text,limit)
        elif self.return_summary_useit:
            if self.return_summary_header_to_text:
                data = self.generate_summary_response(text,limit)
            elif self.return_summary_header_topic_to_text:
                data = self.generate_summary_response(text,limit)
        elif self.return_docs_using_question_answer_useit:
            if self.return_docs_using_question_answer_header_to_text_then_qa:
                data = self.top_similar_docs_qa(text,limit)
            elif self.return_docs_using_question_answer_header_topic_to_text_then_qa:
                data = self.top_similar_docs_qa(text,limit)
        else:
            data = self.top_similar_docs(text,limit)
        
        # add to feedback-loop table
        self.feedback_loop_obj.add_feedback_loop_data({'api':'document_retriever',
            'question':text_bkp,
            'answer':'',
            'other':data,
            'flag':0})

        return data
    
    def get_top_duplicate_records(self,text):
        '''
        given input : text
        1. search in qdrant vector db, get top N records.
        '''
        limit = self.retrieval_docs_limit
        text = text.lower()

        retrieved_data = self.doc_retriever.get_similar_docs_qa(text,limit)

        return retrieved_data
    
    def extract_answer_from_question_given_context(self,question,context):
        response = self.question_answer_model.get_answer_given_question_context(question,context)

        return response
    
    def extract_answer_from_question_given_tablecontext(self,question,table):
        response = self.question_answer_model.get_answer_given_question_table(question,table)

        return response

