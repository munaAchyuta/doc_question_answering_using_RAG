import glob
import json
import os
import time

from ..base.base import BaseClass


class DocRetriver(BaseClass):
    '''
    document retriever.
    '''
    def __init__(self) -> None:
        super().__init__()
        if self.doc_retrieval_config['use_db']['qdrant']:
            from .qdrant_information_retriever import DocRetriver as QdrantDocRetriever
            self.doc_retriever = QdrantDocRetriever()
        elif self.doc_retrieval_config['use_db']['solr']:
            from .solr_information_retriever import DocRetriver as SolrDocRetriever
            self.doc_retriever = SolrDocRetriever()



if __name__ == '__main__':
    docprocessor = DocRetriver()
    docprocessor_instance = docprocessor.doc_retriever
    docprocessor_instance.process_data()

    matched_topic_records = docprocessor_instance.get_similar_topic_docs("give me former employer details from CONTINGENT WORKER CONFIDENTIALITY AGREEMENT ?")
    print(matched_topic_records)
    print("\n===========\n")
    (match_found,topic_match_found) = docprocessor_instance.get_similar_docs("give me former employer details from CONTINGENT WORKER CONFIDENTIALITY AGREEMENT ?")
    print(match_found)

