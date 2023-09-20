import traceback

from ..document_processor.read_pdf_file import DocGenerator
from ..document_processor.pdf_header_generator import PdfHeaderGenerator



class DocsForDb():
    def __init__(self) -> None:
        self.doc_generator = DocGenerator()
        self.header_generator = PdfHeaderGenerator()
    
    def get_structured_docs_for_db(self,file_path,max_seq_length=150,overlap=0,**kwargs):
        kwargs_dict = kwargs['kwargs']

        try:
            if file_path.lower().endswith('.pdf'):
                # generate documents
                document_list,keyword_output = self.doc_generator.generate_doc(file_path,
                                                                                max_seq_length,
                                                                                overlap,
                                                                                kwargs=kwargs_dict)

                # header generator
                headers_dict = self.header_generator.get_headers_with_content(file_path,max_seq_length)

                # add headers to keyword_output/topics
                for each_topic_page in keyword_output:
                    if headers_dict.get(each_topic_page['page_number'],None):
                        each_topic_page['keywords_text'] = each_topic_page['keywords_text'] +', '+ ', '.join(headers_dict[each_topic_page['page_number']]['headers'])
            elif file_path.lower().endswith(('.html', '.json')):
                raise Exception("html/json file processing is not handled yet.")
            elif file_path.lower().endswith(('.docx', '.pptx')):
                raise Exception("docx/pptx file processing is not handled yet.")
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                raise Exception("image file processing is not handled yet.")
            else:
                raise Exception("other file extension processing is not handled yet.")
        except Exception as err:
            print(traceback.format_exc())
            print(f"ERROR:- failed processing file: {file_path}")
        
        return (document_list,keyword_output)
    
    def get_structured_docs_for_db_multicore(self,args):
        file_path = args[0]
        max_seq_length = args[1]
        overlap = args[2]
        kwargs_dict = args[3]
        document_list = []
        keyword_output = []
        file_processed = False
        try:
            if file_path.lower().endswith('.pdf'):
                # generate documents
                document_list,keyword_output = self.doc_generator.generate_doc(file_path,
                                                                                max_seq_length,
                                                                                overlap,
                                                                                kwargs=kwargs_dict)

                # header generator
                headers_dict = self.header_generator.get_headers_with_content(file_path,max_seq_length)

                # add headers to keyword_output/topics
                for each_topic_page in keyword_output:
                    if headers_dict.get(each_topic_page['page_number'],None):
                        each_topic_page['keywords_text'] = each_topic_page['keywords_text'] +', '+ ', '.join(headers_dict[each_topic_page['page_number']]['headers'])
                
                file_processed = True
            elif file_path.lower().endswith(('.html', '.json')):
                raise Exception("html/json file processing is not handled yet.")
            elif file_path.lower().endswith(('.docx', '.pptx')):
                raise Exception("docx/pptx file processing is not handled yet.")
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                raise Exception("image file processing is not handled yet.")
            else:
                raise Exception("other file extension processing is not handled yet.")
        except Exception as err:
            print(traceback.format_exc())
            print(f"ERROR:- failed processing file: {file_path}")
        
        return (file_path,file_processed,document_list,keyword_output)