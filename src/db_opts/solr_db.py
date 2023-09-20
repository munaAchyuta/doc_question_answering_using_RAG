import sys
from typing import List
import requests
import pysolr

from .model_loader import ModelLoader
from ..base.feedback_loop import SqliteDbOpts,JsonFileDrProcessedSqliteDataOpts




class SolrDocRetriever:
    '''
    REFERENCE:
        https://sease.io/2023/01/apache-solr-neural-search-tutorial.html
        https://medium.com/@maithri.vm/from-keywords-to-meaning-embracing-semantic-fusion-in-apache-solrs-hybrid-search-paradigm-e7be29534ddd
        https://github.com/django-haystack/pysolr

    '''
    client = {}
    def __init__(self,solr_address=None,
                 collection_name=None,
                 encoder_model='bert-base-uncased',
                 use_lower_embedding_size=False,
                 window_size=5) -> None:
        self.window_size = window_size

        if solr_address is not None:
            self.SOLR_ADDRESS = solr_address
        else:
            self.SOLR_ADDRESS = 'http://localhost:8983/solr/ms-marco'
        
        self.BATCH_SIZE = 100
        
        self.client_connection()

        self.processed_qna_data_opts = JsonFileDrProcessedSqliteDataOpts("./feedback_loop/processed_solr_records.json")
        self.processed_qna_data_opts.create_file()
        self.processed_qna_data = self.processed_qna_data_opts.read_data()

        # REFERENCE: https://www.sbert.net/docs/training/overview.html
        model_loader = ModelLoader()
        self.encoder = model_loader.get_encoder(model_name=encoder_model,use_dense=use_lower_embedding_size)

        #if collection_name is not None:
        #    self.collection_name = collection_name
        #else:
        #    self.collection_name = 'my_books'

        #max_records = self.check_if_collection_exist()
        #if max_records is None:
        #    self.create_collection()
    
    def client_connection(self,):
        if not SolrDocRetriever.client.get(self.SOLR_ADDRESS,None):
            SolrDocRetriever.client[self.SOLR_ADDRESS] = pysolr.Solr(self.SOLR_ADDRESS, always_commit=True)
        self.client = SolrDocRetriever.client[self.SOLR_ADDRESS]
    
    def create_collection(self,):
        # Create collection to store books
        #self.client.create_collection(
        #    collection_name=self.collection_name,
        #    vectors_config=models.VectorParams(
        #        size=self.encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        #        distance=models.Distance.COSINE
        #    )
        #)
        pass
    
    def check_if_collection_exist(self,):
        #max_index = None
        #try:
        #    records = self.client.get_collection(self.collection_name)
        #    max_index = records.vectors_count
        #    print(f"record counts: {max_index}")
        #except Exception as err:
        #    print(err)
        #    #raise Exception(err)
        #return max_index
        pass
    
    def upload_data_records(self,key="page_content",documents=[]):
        '''
        upload records.
        '''
        #max_index = self.check_if_collection_exist()
        # read json, get max_id
        max_index = self.processed_qna_data.get('max_index',0)
        
        documents_local = []
        # For each document creates a JSON document including both text and related vector. 
        for index, document in enumerate(documents):
            index = index+(max_index+1)
            doc = {
                "id": str(index),
                "payload": document,
                "vector": self.encoder.encode(document[key]).tolist()
                }
            # Append JSON document to a list.
            documents_local.append(doc)

            # To index batches of documents at a time.
            if index % self.BATCH_SIZE == 0 and index != 0:
                # How you'd index data to Solr.
                self.client.add(documents_local)
                documents_local = []
                print("==== indexed {} documents ======".format(index))
        # To index the rest, when 'documents' list < BATCH_SIZE.
        if documents_local:
            self.client.add(documents_local)
        
        # add to processed files
        self.processed_qna_data['max_index'] = index
        self.processed_qna_data_opts.add_data(self.processed_qna_data)
        
        self.client.commit()
        print("finished")

    def search_using_requests(self,input_text='',limit=3):
        '''
        curl -X POST http://localhost:8983/solr/ms-marco/select?fl=id,text,score -d '
        {
        "query": "{!knn f=vector topK=3}[-9.01364535e-03, -7.26634488e-02, -1.73818860e-02, ..., -1.16323479e-01]"
        }'
        '''
        url = f"{self.SOLR_ADDRESS}/select?fl=id,payload,score"
        headers={'Content-type':'application/json'}
        payload = {
                "query": "{!knn f=vector topK=limit}{vector_arr}".format(vector_arr=self.encoder.encode(input_text).tolist(),limit=limit)
                }
        response = requests.post(url, headers=headers, data=payload, timeout=5)

        return response
    
    def get_all_records(self,):
        #retrieve all the recrods from collection
        solr_response = self.client.search(
                            q='*:*',
                            rows=30,
                            start=0,
                            fl='id,title',
                            wt='json')
        
        return solr_response
    
    def keyword_search(self,input_text='',limit=5):
        print(f"performing keyword search for {input_text}")
        solr_response = self.client.search(
                            q=input_text,
                            rows=limit,
                            start=0,
                            fl='node-uuid,title,score,category',
                            qf='title^5 description^3',
                            pf='title^5',
                            defType='edismax',
                            ps='7',
                            wt='json')
        
        return solr_response
    
    def get_records(self,input_text='',limit=5):
        class MyObject:
            def __init__(self, d=None):
                if d is not None:
                    for key, value in d.items():
                        setattr(self, key, value)
            def dict(self,):
                return self.__dict__
        # example
        solr_response = {
                "responseHeader":{
                "response":{"numFound":3,"start":0,"maxScore":0.44739443,"numFoundExact":True,"docs":[
                    {
                        "id":7686,
                        "file_name":"Test",
                        "page_number":1,
                        "page_content":"A. A federal tax identification number ... to identify your business to several federal agencies responsible for the regulation of business.\n",
                        "score":0.44739443},
                    {
                        "id":7691,
                        "file_name":"Test",
                        "page_number":1,
                        "page_content":"A. A federal tax identification number (also known as an employer identification number or EIN), is a number assigned solely to your business by the IRS.\n",
                        "score":0.44169965},
                    {
                        "id":7692,
                        "file_name":"Test",
                        "page_number":1,
                        "page_content":"Letâs start at the beginning. A tax ID number or employer identification number (EIN) is a number ... to a business, much like a social security number does for a person.\n",
                        "score":0.43761322}]
                }
                }
            }
        solr_response = [MyObject(i) for i in solr_response["responseHeader"]["response"]["docs"]]
        return solr_response
        solr_response = self.client.search(
            fl=['id','payload','score'],
            q="{!knn f=vector topK=limit}".format(limit=limit) + str(self.encoder.encode(input_text).tolist()),
            rows = limit
            )
        
        return solr_response


if __name__ == "__main__":
    documents = [
                    {
                    "page_content": "Harness the power of storytelling to influence and persuade others effectively.",
                    "file_name": "Article",
                    "page_number": 1
                    }
                ]

    solr_doc_obj = SolrDocRetriever()
    _ = solr_doc_obj.upload_data_records(documents=documents)
    hits = solr_doc_obj.get_records_dict("Unspoken language")