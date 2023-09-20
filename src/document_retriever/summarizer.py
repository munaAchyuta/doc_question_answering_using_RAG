import itertools
from collections import defaultdict
import requests
import torch
torch.manual_seed(45)
from transformers import AutoModelWithLMHead, AutoTokenizer

from ..base.base import BaseClass
from ..templates.doc_summarizer_template import template, template_2

class Summarizer(BaseClass):
    '''
    Summarizer class for summarizing document(derived using similarity metric), 
    given input text/question.
    '''
    def __init__(self) -> None:
        super().__init__()
        self.uset_t5 = self.doc_retrieval_config['document_retrieval_structure']['return_summary']['use_t5']
        self.t5_model_name = self.doc_retrieval_config['document_retrieval_structure']['return_summary']['t5_model']
        self.uset_openai = self.doc_retrieval_config['document_retrieval_structure']['return_summary']['use_openai']
        self.use_sort_of_sent_from_text = self.doc_retrieval_config['document_retrieval_structure']['return_summary']['use_sort_of_sent_from_text']
        self.use_sort_of_sent_from_text_useit = self.use_sort_of_sent_from_text['use_it']
        self.use_sort_of_sent_from_text_chunk_size = self.use_sort_of_sent_from_text['chunk_size']

        self.t5_tokenizer = AutoTokenizer.from_pretrained(self.t5_model_name)
        self.t5_model = AutoModelWithLMHead.from_pretrained(self.t5_model_name)

    def t5_summarize(self,text, max_length=150):
        input_ids = self.t5_tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

        generated_ids = self.t5_model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)

        preds = [self.t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

        return preds[0]

    def call_openai_api(self, input_prompt):
        sql = ""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_token}"
        }
        body = {
            "model": self.openai_model,
            "prompt": input_prompt,
            "max_tokens": self.openai_max_token,
            "temperature": self.openai_temperature
        }
        out = requests.post(self.openai_url, json=body, headers=headers).json()
        if out.get('choices', None):
            sql = out['choices'][0]['text'].strip('\n')

        return sql
    
    def summarize(self,document_text,input_text,dont_check_again=False):
        if self.uset_openai:
            input_prompt = template.format(input_text=input_text,document_text=document_text)
            self.logger.info(f"input_prompt: {input_prompt}")

            # call OpenAI API
            response = self.call_openai_api(input_prompt)
        elif self.uset_t5:
            #raise Exception("T5 module is not defined.")
            response = self.t5_summarize(document_text)

        '''
        # call again
        if not dont_check_again:
            print('called...........')
            if (response.strip().replace('.','').replace(',','').lower() == 'no') \
                or (len(response)==0) or (any([True for i in ['no','no.','no,','answer: no'] if i in response.strip().replace('.','').replace(',','').lower()])):
                response = self.ask_again()
                if response.strip().replace('.','').replace(',','').lower() != 'no':
                    dont_check_again = True
        '''
        return response
    
    def ask_again(self):
        input_prompt = template_2
        self.logger.info(f"input_prompt: {input_prompt}")

        # call OpenAI API
        response = self.call_openai_api(input_prompt)
        self.logger.info(f"Doc Summarizer response again: {response}")

        return response