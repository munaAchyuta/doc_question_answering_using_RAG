import re
from io import StringIO
import traceback

from PyPDF2 import PdfReader
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import HTMLConverter,TextConverter,XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import yake
import requests
import tabula 
from tabulate import tabulate
import nltk
import fitz

from .pdf_extract_text_without_table import DocPdfplumber
from .pdf_extract_text_with_image import get_text_from_image

class DocGenerator:
    def __init__(self) -> None:
        self.kw_extractor = yake.KeywordExtractor(top=20,stopwords=None)
        self.max_seq_length = 700
        self.template = '''here is the document,

                        document: """{input_document}"""

                        question: "what is the document about ?"
                        '''
    
    def call_openai_api(self, input_prompt, **kwargs):
        response = ""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {kwargs['kwargs']['openai_token']}"
        }
        body = {
            "model": kwargs['kwargs']['openai_model'],
            "prompt": input_prompt,
            "max_tokens": kwargs['kwargs']['openai_max_token'],
            "temperature": kwargs['kwargs']['openai_temperature']
        }
        out = requests.post(kwargs['kwargs']['openai_url'], json=body, headers=headers).json()
        if out.get('choices', None):
            response = out['choices'][0]['text'].strip('\n')

        return response
    
    def get_table(self,pdf_file,page_num):
        '''
        given input pdf_file and page_number
        return tuple having (True if table exists else False, list of tables)
        '''
        tables = tabula.read_pdf(pdf_file,pages=page_num,multiple_tables=True)
        if len(tables) == 0:
            return (False,tables)
        
        return (True,tables)
    
    def check_if_file_extractable_or_not(self,path):
        doc = fitz.open(path)
        data = []
        for page in doc:
            data.append(len(page.get_text()))
        
        if len([True for i in data if i==0]) == len(data):
            return False
        
        return True
    
    def read_pdf_nonextractable_old(self,path,max_seq_length=150,**kwargs):
        import pytesseract
        from pdf2image import convert_from_path

        pages = convert_from_path(path, 500)

        data = dict()
        data['file_name'] = path
        data['content'] = list()
        for pageNum,imgBlob in enumerate(pages):
            page_data = dict()
            page_data['page_no'] = pageNum+1
            page_data['page_content'] = list()
            page_data['key_words'] = list()
            page_data['page_summary'] = ""
            page_data['raw_data'] = ""

            text = pytesseract.image_to_string(imgBlob,lang='eng')

            page_data['raw_data'] = text[:]
            text_bkp = text[:]
            text = self.text_cleansing(text)

            # extract key phrases
            if kwargs['kwargs'].get('topics',None):
                keywords = self.kw_extractor.extract_keywords(text)
                keywords = [kw for kw,v in keywords]
                page_data['key_words'].extend(keywords)
            
            # summary of data
            if kwargs['kwargs'].get('data_summary',None):
                inp_prompt = self.template.format(input_document=text_bkp)
                data_summary = self.call_openai_api(input_prompt=inp_prompt,**kwargs)
                #page_data['page_summary'] = data_summary
                page_data['key_words'] = [data_summary]

            # slice documents in small chunk of size self.document_length
            chunked_text = DocGenerator.get_sliced_text(text,max_seq_length)
            page_data['page_content'].extend(chunked_text)

            # table extraction if present
            (table_present,tables) = self.get_table(path,pageNum+1)
            page_data['table_present'] = table_present
            page_data['table'] = tables
            
            data['content'].append(page_data)

        return data

    def read_pdf_old(self,path,max_seq_length=150,overlap=0,**kwargs):
        if not self.check_if_file_extractable_or_not(path):
            raise Exception(f"file: {path} , not extractable.")
        
        pdf = PdfReader(open(path, "rb"))
        fp = open(path, 'rb')
        num_of_pages = pdf._get_num_pages()
        data = dict()
        data['file_name'] = path
        data['content'] = list()
        for page_i in range(num_of_pages):
            page_data = dict()
            page_data['page_no'] = page_i+1
            inside = [page_i]
            pagenos=set(inside)
            rsrcmgr = PDFResourceManager()
            retstr = StringIO()
            #codec = 'utf-8'
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr, laparams=laparams)#, codec=codec
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ""
            maxpages = 0
            caching = True
            text = ""
            page_data['page_content'] = list()
            page_data['key_words'] = list()
            page_data['page_summary'] = ""
            page_data['raw_data'] = ""
            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
                interpreter.process_page(page)
                text = retstr.getvalue()
                #text = text.decode("ascii","replace")
                page_data['raw_data'] = text[:]
                text_bkp = text[:]
                text = self.text_cleansing(text)

                # extract key phrases
                if kwargs['kwargs'].get('topics',None):
                    keywords = self.kw_extractor.extract_keywords(text)
                    keywords = [kw for kw,v in keywords]
                    page_data['key_words'].extend(keywords)
                
                # summary of data
                if kwargs['kwargs'].get('data_summary',None):
                    inp_prompt = self.template.format(input_document=text_bkp)
                    data_summary = self.call_openai_api(input_prompt=inp_prompt,**kwargs)
                    #page_data['page_summary'] = data_summary
                    page_data['key_words'] = [data_summary]

                # slice documents in small chunk of size self.document_length
                chunked_text = DocGenerator.get_sliced_text(text,max_seq_length,overlap)
                page_data['page_content'].extend(chunked_text)
            
            # table extraction if present
            (table_present,tables) = self.get_table(path,page_i+1)
            page_data['table_present'] = table_present
            page_data['table'] = tables
            
            data['content'].append(page_data)

        return data
    
    def read_pdf_nonextractable(self,path,max_seq_length=150,**kwargs):
        pages_data = get_text_from_image(path)
        data = dict()
        data['file_name'] = path
        data['content'] = list()
        for page_dict in pages_data:
            page_data = dict()
            page_data['page_no'] = page_dict['page_number']+1
            page_data['page_content'] = list()
            page_data['key_words'] = list()
            page_data['page_summary'] = ""
            page_data['raw_data'] = ""

            text = page_dict['page_content']

            page_data['raw_data'] = text[:]
            text_bkp = text[:]
            text = self.text_cleansing(text)

            # extract key phrases
            if kwargs['kwargs'].get('topics',None):
                keywords = self.kw_extractor.extract_keywords(text)
                keywords = [kw for kw,v in keywords]
                page_data['key_words'].extend(keywords)
            
            # summary of data
            if kwargs['kwargs'].get('data_summary',None):
                inp_prompt = self.template.format(input_document=text_bkp)
                data_summary = self.call_openai_api(input_prompt=inp_prompt,**kwargs)
                #page_data['page_summary'] = data_summary
                page_data['key_words'] = [data_summary]

            # slice documents in small chunk of size self.document_length
            #chunked_text = DocGenerator.get_sliced_text(text,max_seq_length)
            #page_data['page_content'].extend(chunked_text)

            # new code.
            chunked_text = DocGenerator.get_sliced_text(text,max_seq_length)
            page_data['page_content'].extend([{"text":text_chunk,"is_table":False} for text_chunk in chunked_text])

            # table extraction if present
            #(table_present,tables) = self.get_table(path,pageNum+1)
            page_data['table_present'] = False
            page_data['table'] = ""
            
            data['content'].append(page_data)

        return data

    def read_pdf(self,path,max_seq_length=150,overlap=0,**kwargs):
        #===== used for separating table from text/paragraph from a pdf page.
        data_pages = None
        try:
            doc_pdfplumber = DocPdfplumber(path)
            data_pages = doc_pdfplumber.get_pdf_page_info()
        except Exception as err:
            print(traceback.format_exc())
        #=====

        pdf = PdfReader(open(path, "rb"))
        fp = open(path, 'rb')
        num_of_pages = pdf._get_num_pages()
        data = dict()
        data['file_name'] = path
        data['content'] = list()
        for page_i in range(num_of_pages):
            page_data = dict()
            page_data['page_no'] = page_i+1
            inside = [page_i]
            pagenos=set(inside)
            rsrcmgr = PDFResourceManager()
            retstr = StringIO()
            #codec = 'utf-8'
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr, laparams=laparams)#, codec=codec
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ""
            maxpages = 0
            caching = True
            text = ""
            page_data['page_content'] = list()
            page_data['key_words'] = list()
            page_data['page_summary'] = ""
            page_data['raw_data'] = ""
            page_data['table_present_chunk'] = False
            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
                interpreter.process_page(page)
                text = retstr.getvalue()
                #text = text.decode("ascii","replace")
                text_bkp = text[:]
                text = self.text_cleansing(text)
                page_data['raw_data'] = text[:]

                # extract key phrases
                if kwargs['kwargs'].get('topics',None):
                    keywords = self.kw_extractor.extract_keywords(text)
                    keywords = [kw for kw,v in keywords]
                    page_data['key_words'].extend(keywords)
                
                # summary of data
                if kwargs['kwargs'].get('data_summary',None):
                    inp_prompt = self.template.format(input_document=text_bkp)
                    data_summary = self.call_openai_api(input_prompt=inp_prompt,**kwargs)
                    #page_data['page_summary'] = data_summary
                    page_data['key_words'] = [data_summary]

                # slice documents in small chunk of size self.document_length
                #chunked_text = DocGenerator.get_sliced_text(text,max_seq_length)
                #page_data['page_content'].extend(chunked_text)

                # new code. for separating table and text out of whole page.
                if data_pages is None:
                    print(f"{path} : pdfplumber gave error. so using pdfminer.")
                    chunked_text = DocGenerator.get_sliced_text(text,max_seq_length)
                    page_data['page_content'].extend([{"text":text_chunk,"is_table":False} for text_chunk in chunked_text])
                else:
                    chunked_text = DocGenerator.get_sliced_text(data_pages[page_i]['page_without_table'],max_seq_length)
                    page_data['page_content'].extend([{"text":text_chunk,"is_table":False} for text_chunk in chunked_text])
                    if data_pages[page_i]['table_present']:
                        chunked_text = DocGenerator.get_sliced_text(data_pages[page_i]['tables_text'],max_seq_length)
                        page_data['page_content'].extend([{"text":text_chunk,"is_table":True} for text_chunk in chunked_text])
            
            # table extraction if present
            (table_present,tables) = self.get_table(path,page_i+1)
            page_data['table_present'] = table_present
            page_data['table'] = tables
            
            data['content'].append(page_data)

        return data
    
    def text_cleansing(self,text):
        text = text.replace("\n"," ").lower()
        text = ' '.join(text.split())

        return text
    
    @staticmethod
    def get_index_slicer_range(size=0,window=100,overlap=0):
        prev = 0
        output = []
        for j in [i for i in range(window,size+window,window)]:
            output.append((prev,j))
            prev = j-overlap
        
        return output

    @staticmethod
    def get_sliced_text(text,max_seq_length,overlap=0):
        '''
        purpose:
            slice text into chunks.
        rules:
            if overlap==0 means use sentence tokenizer to chunk text.
            else default text chunker.
        '''
        if overlap==0:
            sliced_text = []
            text_sents = nltk.sent_tokenize(text)
            for each_sent in text_sents:
                if len(each_sent) > max_seq_length:
                    index_slicer = DocGenerator.get_index_slicer_range(len(each_sent),max_seq_length,overlap)
                    sliced_each_sent = [each_sent[i:j] for i,j in index_slicer]
                    sliced_text.extend(sliced_each_sent)
                else:
                    sliced_text.append(each_sent)
        else:
            index_slicer = DocGenerator.get_index_slicer_range(len(text),max_seq_length,overlap)
            sliced_text = [text[i:j] for i,j in index_slicer]

        return sliced_text
    
    def generate_doc_old(self,path=None,max_seq_length=150,overlap=0,**kwargs):
        '''
        DB records keys defined here.
        it's always good practice to have fixed keys throughout all collections.
        because it'll help later operation easier.
        '''
        data = self.read_pdf(path,max_seq_length,overlap,**kwargs)
        file_name = data['file_name']
        content = data['content']
        new_data = []
        data_with_keywords = []
        for each_page_content in content:
            tmp_keywords = dict()
            tmp_keywords['file_name'] = file_name
            tmp_keywords['page_number'] = each_page_content['page_no']
            tmp_keywords['raw_data'] = each_page_content['raw_data']
            tmp_keywords['table_present'] = each_page_content['table_present']
            tmp_keywords['table'] = each_page_content['table']
            tmp_keywords['keywords'] = each_page_content['key_words']
            tmp_keywords['keywords_text'] = f"{file_name} having key phrases are "+", ".join(each_page_content['key_words'])
            data_with_keywords.append(tmp_keywords)

            for id,small_content in enumerate(each_page_content['page_content']):
                tmp = dict()
                tmp['file_name'] = file_name
                tmp['page_number'] = each_page_content['page_no']
                tmp['table_present'] = each_page_content['table_present']
                tmp['page_content'] = small_content
                tmp['seq_id'] = id
                new_data.append(tmp)
        
        return (new_data,data_with_keywords)
    
    def generate_doc(self,path=None,max_seq_length=150,overlap=0,**kwargs):
        if not self.check_if_file_extractable_or_not(path):
            #raise Exception(f"file: {path} , not extractable.")
            print((f"file: {path} , not extractable."))
            data = self.read_pdf_nonextractable(path,max_seq_length,**kwargs)
        else:
            print((f"file: {path} , extractable."))
            data = self.read_pdf(path,max_seq_length,**kwargs)
        
        file_name = data['file_name']
        content = data['content']
        new_data = []
        data_with_keywords = []
        for each_page_content in content:
            tmp_keywords = dict()
            tmp_keywords['file_name'] = file_name
            tmp_keywords['page_number'] = each_page_content['page_no']
            tmp_keywords['raw_data'] = each_page_content['raw_data']
            tmp_keywords['table_present'] = each_page_content['table_present']
            tmp_keywords['table'] = each_page_content['table']
            tmp_keywords['keywords'] = each_page_content['key_words']
            tmp_keywords['keywords_text'] = f"{file_name} having key phrases are "+", ".join(each_page_content['key_words'])
            data_with_keywords.append(tmp_keywords)
            '''
            for id,small_content in enumerate(each_page_content['page_content']):
                tmp = dict()
                tmp['file_name'] = file_name
                tmp['page_number'] = each_page_content['page_no']
                tmp['table_present'] = each_page_content['table_present']
                #tmp['page_content'] = small_content
                tmp['page_content'] = small_content['text']
                tmp['is_table'] = small_content['is_table']
                tmp['seq_id'] = id
                new_data.append(tmp)
            '''
            
            for id,small_content in enumerate(each_page_content['page_content']):
                tmp = dict()
                tmp['file_name'] = file_name
                tmp['page_number'] = each_page_content['page_no']
                tmp['table_present'] = each_page_content['table_present']
                tmp['page_content'] = small_content['text']
                tmp['is_table'] = small_content['is_table']
                tmp['seq_id'] = id
                new_data.append(tmp)
        
        return (new_data,data_with_keywords)



if __name__ == '__main__':
    path = "data/pdf_docs/AMIND-Contingent Worker Agreement _ revised (1).pdf"

    doc_generator = DocGenerator()
    output, keyword_output = doc_generator.generate_doc(path)
    print(output[:5])
    print("==============")
    print(output[-5:])
    print("--------------")
    print(keyword_output)