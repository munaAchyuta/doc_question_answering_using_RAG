import numpy as np
import torch
torch.manual_seed(45)
from sentence_transformers import models as st_models
from sentence_transformers import SentenceTransformer, util
from torch import nn
from transformers import pipeline
from dataclasses import dataclass

from ..base.base import BaseClass

@dataclass
class ModelLoader():
    model_catched = dict()
    logger = BaseClass().logger
    
    def get_encoder(self,model_name='bert-base-uncased',use_dense=False):
        '''
        get model to generate embeddings.
        '''
        if ModelLoader.model_catched.get(model_name,None):
            self.logger.info(f"using cached model: {model_name}")
            model = ModelLoader.model_catched.get(model_name)
            return model
        
        if use_dense:
            word_embedding_model = st_models.Transformer(model_name, max_seq_length=256)
            pooling_model = st_models.Pooling(word_embedding_model.get_word_embedding_dimension())
            dense_model = st_models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
        else:
            model = SentenceTransformer(model_name)
        
        ModelLoader.model_catched[model_name] = model

        return model
    
    def get_qa_model(self,task_name='question-answering',model_name='distilbert-base-cased-distilled-squad'):
        '''
        get model which generate answer from given question & context.
        '''
        if ModelLoader.model_catched.get(model_name,None):
            self.logger.info(f"using qa cached model: {model_name}")
            model = ModelLoader.model_catched.get(model_name)
            return model
        
        model = pipeline(task=task_name,model=model_name)

        ModelLoader.model_catched[model_name] = model

        return model


if __name__ == '__main__':
    model_loader = ModelLoader()
    model = model_loader.get_encoder(model_name='all-MiniLM-L6-v2')