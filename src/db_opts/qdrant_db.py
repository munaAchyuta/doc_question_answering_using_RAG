import numpy as np
import torch
torch.manual_seed(45)
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as st_models
from torch import nn
from .model_loader import ModelLoader

#======================== patching qdrant_client library module to handle multi-threading issue.
import qdrant_client
import sqlite3
import wrapt
@wrapt.patch_function_wrapper(qdrant_client.local.persistence.CollectionPersistence, '__init__')
def new_init(wrapped, instance, args, kwargs):
    # here, wrapped is the original __init__,
    # instance is `self` instance (it is not true for classmethods though),
    # args and kwargs are tuple and dict respectively.

    # first call original init
    wrapped(*args, **kwargs)  # note it is already bound to the instance
    # and now do our changes
    instance.storage = sqlite3.connect(str(instance.location), check_same_thread=False)
    instance._ensure_table()

from qdrant_client import models, QdrantClient
from qdrant_client.models import PointStruct
#======================== patching end.



class QdrantDocRetriever:
    client = {}
    def __init__(self,vector_db_path=None,
                 collection_name=None,
                 encoder_model='bert-base-uncased',
                 use_lower_embedding_size=False,
                 window_size=5) -> None:
        self.window_size = window_size

        if vector_db_path is not None:
            self.vector_db_path = vector_db_path
        else:
            self.vector_db_path = "./data/docs"
        
        self.client_connection()

        # REFERENCE: https://www.sbert.net/docs/training/overview.html
        model_loader = ModelLoader()
        self.encoder = model_loader.get_encoder(model_name=encoder_model,use_dense=use_lower_embedding_size)

        if collection_name is not None:
            self.collection_name = collection_name
        else:
            self.collection_name = 'my_books'

        max_records = self.check_if_collection_exist()
        if max_records is None:
            self.create_collection()
    
    def client_connection(self,):
        if not QdrantDocRetriever.client.get(self.vector_db_path,None):
            QdrantDocRetriever.client[self.vector_db_path] = QdrantClient(path=self.vector_db_path) # Persists changes to disk, fast prototyping
        self.client = QdrantDocRetriever.client[self.vector_db_path]
    
    def create_collection(self,):
        # Create collection to store books
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
                distance=models.Distance.COSINE
            )
        )
    
    def recreate_collection(self,):
        # re-create collection to store books
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
                distance=models.Distance.COSINE
            )
        )
    
    def check_if_collection_exist(self,):
        max_index = None
        try:
            records = self.client.get_collection(self.collection_name)
            max_index = records.vectors_count
            print(f"record counts: {max_index}")
        except Exception as err:
            print(err)
            #raise Exception(err)
        
        return max_index

    def upload_data_records(self,key="page_content",documents=[]):
        max_index = self.check_if_collection_exist()
        
        # Let's vectorize descriptions and upload to qdrant
        #collection = self.client._get_collection(self.collection_name)
        self.client.upload_records(
            collection_name=self.collection_name,
            records=[
                models.Record(
                    id=idx+(max_index+1),
                    vector=self.encoder.encode(doc[key]).tolist(),
                    payload=doc
                ) for idx, doc in enumerate(documents)
            ]
        )
    
    @staticmethod
    def get_spacy_model():
        '''
        Function to get spacy model with medium neural net.
        Args: None
        Returns:
            nlp: A loaded model with constituency parsing functionality.
        '''
        import spacy
        nlp = spacy.load('en_core_web_md')
        print("Model loaded")
        return nlp
    
    def word_piece_tokens_avg(self,text):
        '''
        input: toekn_embd = self.encoder.encode('Hello there Achyuta.',output_value='token_embeddings')
        '''
        toekn_embd = self.encoder.encode(text,output_value='token_embeddings')
        return np.mean(toekn_embd.numpy(), axis=0)
    
    @staticmethod
    def wva(string):
        '''
        Finds document vector through an average of each word's vector.
        Args: string (str): Input sentence
        Returns: array: Word vector average
        '''
        nlp = QdrantDocRetriever.get_spacy_model()
        doc = nlp(string)
        wvs = np.array([doc[i].vector for i in range(len(doc))])
        return np.mean(wvs, axis=0)

    def get_records(self,input_text='',limit=5):
        # Let's now search for something
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=self.encoder.encode(input_text).tolist(),
            limit=limit
        )

        return hits
    
    def get_records_dict(self,input_text='',limit=5):
        # Let's now search for something
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=self.encoder.encode(input_text).tolist(),
            limit=limit
        )

        hits = [i.dict() for i in hits]
        
        return hits
    
    def get_records_with_filter(self,input_text='',field='',limit=5):
        # Let's now search only for books from 21st century
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=self.encoder.encode(input_text).tolist(),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=field,
                        #range=models.Range(gte=0,lte=8)
                    )
                ]
            ),
            limit=limit
        )
        
        return hits
    
    def get_records_with_filter_dict(self,input_text='',field='',limit=5):
        # Let's now search only for books from 21st century
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=self.encoder.encode(input_text).tolist(),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=field,
                        #range=models.Range(gte=0,lte=8)
                    )
                ]
            ),
            limit=limit
        )

        hits = [i.dict() for i in hits]
        
        return hits
    
    def get_records_window(self,input_text='',limit=5,docs_window=5):
        hits = self.get_records(input_text,limit)
        hits_window = []
        for hit in hits:
            tmp = dict()
            tmp['file_name'] = hit.payload['file_name']
            tmp['page_number'] = hit.payload['page_number']
            nve = [hit.payload['seq_id']-i for i in range(1,docs_window)]
            pve = [hit.payload['seq_id']+i for i in range(0,docs_window)]
            tmp['seq_id'] = nve + pve
            hits_window.append(tmp)
        
        hits_window_tmp = []
        for item in hits_window:
            found = False
            for output_item in hits_window_tmp:
                if (
                    output_item['file_name'] == item['file_name']
                    and output_item['page_number'] == item['page_number']
                ):
                    output_item['seq_id'].extend(item['seq_id'])
                    found = True
                    break
            if not found:
                hits_window_tmp.append(item)
        
        windowed_output = self.get_records_with_multi_filter(input_text,hits_window_tmp,limit=75)

        return windowed_output
    
    def add_field_conditions(self,field_conditions=[]):
        cond_list = []
        for each_cond in field_conditions:
            inner_cond = []
            for key_name in each_cond:
                if isinstance(each_cond[key_name],list):
                    tmp = models.FieldCondition(
                                key=key_name,
                                match=models.MatchAny(any=each_cond[key_name])
                                )
                else:
                    tmp = models.FieldCondition(
                                key=key_name,
                                match={'value':each_cond[key_name]}
                                )
                inner_cond.append(tmp)
            cond_list.append(models.Filter(must=inner_cond))
        
        return cond_list
    
    def get_records_with_multi_filter_direct(self,input_text='',field_conditions=[],limit=15):
        '''
        input:
            field_conditions: [{'file_name':'','page_no':4,'seq_id':[3,4,5]}]
        '''
        cond_list = self.add_field_conditions(field_conditions)

        # Let's now search only for books from 21st century
        hits = self.client.scroll(
            collection_name=self.collection_name,
            #query_vector=self.encoder.encode(input_text).tolist(),
            scroll_filter=models.Filter(
                should=cond_list
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        hits = [i[0] for i in hits if i is not None]

        return hits
    
    def get_records_with_multi_filter(self,input_text='',field_conditions=[],limit=15):
        '''
        input:
            field_conditions: [{'file_name':'','page_no':4,'seq_id':[3,4,5]}]
        '''
        cond_list = self.add_field_conditions(field_conditions)

        # Let's now search only for books from 21st century
        #hits = self.client.search(
        #    collection_name=self.collection_name,
        #    query_vector=self.encoder.encode(input_text).tolist(),
        #    query_filter=models.Filter(
        #        should=cond_list
        #    ),
        #    limit=limit
        #)
        hits = self.client.scroll(
            collection_name=self.collection_name,
            #query_vector=self.encoder.encode(input_text).tolist(),
            scroll_filter=models.Filter(
                should=cond_list
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits = [i[0] for i in hits if i is not None]

        t_hits_dict = [{'score':hit.score,'file_name':hit.payload['file_name'],'page_number':hit.payload['page_number'],'seq_id':hit.payload['seq_id'],'text':hit.payload['page_content']} for hit in hits]
        #return t_hits_dict

        return_list = []
        for each_i in field_conditions:
            trp = [j for j in t_hits_dict if each_i['file_name']==j['file_name'] and each_i['page_number']==j['page_number']]
            trp = sorted(trp,key=lambda x: x['seq_id'])
            tt_text = " ".join([i['text'] for i in trp])
            return_list.append({"file": each_i['file_name'],
                                "page": each_i['page_number'],
                                "seq_ids":[i['seq_id'] for i in trp],
                                "text":tt_text
                                })
        
        return return_list


if __name__ == '__main__':
    # Let's make a semantic search for Sci-Fi books! 
    documents = [
    { "name": "The Time Machine", "page_content": "A man travels through time and witnesses the evolution of humanity.", "author": "H.G. Wells", "year": 1895 },
    { "name": "Ender's Game", "page_content": "A young boy is trained to become a military leader in a war against an alien race.", "author": "Orson Scott Card", "year": 1985 },
    { "name": "Brave New World", "page_content": "A dystopian society where people are genetically engineered and conditioned to conform to a strict social hierarchy.", "author": "Aldous Huxley", "year": 1932 },
    { "name": "The Hitchhiker's Guide to the Galaxy", "page_content": "A comedic science fiction series following the misadventures of an unwitting human and his alien friend.", "author": "Douglas Adams", "year": 1979 },
    { "name": "Dune", "page_content": "A desert planet is the site of political intrigue and power struggles.", "author": "Frank Herbert", "year": 1965 },
    { "name": "Foundation", "page_content": "A mathematician develops a science to predict the future of humanity and works to save civilization from collapse.", "author": "Isaac Asimov", "year": 1951 },
    { "name": "Snow Crash", "page_content": "A futuristic world where the internet has evolved into a virtual reality metaverse.", "author": "Neal Stephenson", "year": 1992 },
    { "name": "Neuromancer", "page_content": "A hacker is hired to pull off a near-impossible hack and gets pulled into a web of intrigue.", "author": "William Gibson", "year": 1984 },
    { "name": "The War of the Worlds", "page_content": "A Martian invasion of Earth throws humanity into chaos.", "author": "H.G. Wells", "year": 1898 },
    { "name": "The Hunger Games", "page_content": "A dystopian society where teenagers are forced to fight to the death in a televised spectacle.", "author": "Suzanne Collins", "year": 2008 },
    { "name": "The Andromeda Strain", "page_content": "A deadly virus from outer space threatens to wipe out humanity.", "author": "Michael Crichton", "year": 1969 },
    { "name": "The Left Hand of Darkness", "page_content": "A human ambassador is sent to a planet where the inhabitants are genderless and can change gender at will.", "author": "Ursula K. Le Guin", "year": 1969 },
    { "name": "The Time Traveler's Wife", "page_content": "A love story between a man who involuntarily time travels and the woman he loves.", "author": "Audrey Niffenegger", "year": 2003 }
    ]

    documents_next = [
    { "name": "Finance Doc", "page_content": "this one talks about stock market.", "author": "Achyuta", "year": 2023},
    { "name": "Medical Doc", "page_content": "this one talks about medical industry.", "author": "Achyuta", "year": 2023},
    { "name": "Sports Doc", "page_content": "this one talks about sports cricket.", "author": "Achyuta", "year": 2023},
    ]

    # Experiment
    qdrant_obj = QdrantDocRetriever()

    #qdrant_obj.upload_records(documents)

    #qdrant_obj.upload_records(documents_next)

    match_found = qdrant_obj.get_records('give me former employer details from CONTINGENT WORKER CONFIDENTIALITY AGREEMENT ?',5)
    #print("===============")
    #print(match_found)
    match_found = qdrant_obj.get_records_window('give me former employer details from CONTINGENT WORKER CONFIDENTIALITY AGREEMENT ?',5)
    print("===============")
    print(match_found)
    print()
    print("size of document: ",len(match_found))