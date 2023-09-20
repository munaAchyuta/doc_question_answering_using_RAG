import glob
import time
from multiprocessing import Pool
from itertools import product
import multiprocessing

from ..base.base import BaseClass


class DocProcessor(BaseClass):
    '''
    document processor.
    '''
    def __init__(self) -> None:
        super().__init__()
        if self.doc_retrieval_config['use_db']['qdrant']:
            from .qdrant_processor import DocProcessor as QrantProcessor
            self.doc_proc = QrantProcessor()
        elif self.doc_retrieval_config['use_db']['solr']:
            from .solr_processor import DocProcessor as SolrProcessor
            self.doc_proc =  SolrProcessor()
    
    def process_documents(self,):
        if self.doc_retrieval_config['use_db']['qdrant']:
            #docprocessor.process_data()
            total_cores = multiprocessing.cpu_count()
            self.doc_proc.process_data_using_multicores(n_cores=total_cores-1)

            matched_records = self.doc_proc.get_similar_docs("give me former employer details from CONTINGENT WORKER CONFIDENTIALITY AGREEMENT ?")
            print(matched_records)
        elif self.doc_retrieval_config['use_db']['solr']:
            self.doc_proc.process_data()

            matched_records = self.doc_proc.get_similar_docs("give me former employer details from CONTINGENT WORKER CONFIDENTIALITY AGREEMENT ?")
            print(matched_records)

if __name__ == '__main__':
    docprocessor = DocProcessor()
    docprocessor.process_documents()

