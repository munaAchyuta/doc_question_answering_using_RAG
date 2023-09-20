import re 
import numpy as np
import pandas as pd
#from unidecode import unidecode 
import fitz

class PdfHeaderGenerator():
    def __init__(self) -> None:
        pass
    
    def check_if_file_extractable_or_not(self,path):
        doc = fitz.open(path)
        data = []
        for page in doc:
            data.append(len(page.get_text()))
        
        if len([True for i in data if i==0]) == len(data):
            return False
        
        return True
    
    def read_pdf(self,pdf_path):
        self.doc = fitz.open(pdf_path)

    def get_block_dict(self,):
        self.block_dict = {}
        page_num = 1
        for page in self.doc: # Iterate all pages in the document
            file_dict = page.get_text('dict') # Get the page dictionary 
            block = file_dict['blocks'] # Get the block information
            self.block_dict[page_num] = block # Store in block dictionary
            page_num += 1 # Increase the page value by 1
    
    def get_span_df(self,):
        spans = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'text', 'tag'])
        rows = []
        for page_num, blocks in self.block_dict.items():
            for block in blocks:
                if block['type'] == 0:
                    for line in block['lines']:
                        for span in line['spans']:
                            xmin, ymin, xmax, ymax = list(span['bbox'])
                            font_size = span['size']
                            text = span['text']#unidecode(span['text'])
                            span_font = span['font']
                            is_upper = False
                            is_bold = False 
                            if "bold" in span_font.lower():
                                is_bold = True 
                            if re.sub("[\(\[].*?[\)\]]", "", text).isupper():
                                is_upper = True
                            if text.replace(" ","") !=  "":
                                rows.append((page_num,xmin, ymin, xmax, ymax, text, is_upper, is_bold, span_font, font_size))

        self.span_df = pd.DataFrame(rows, columns=['page_num','xmin','ymin','xmax','ymax', 'text', 'is_upper','is_bold','span_font', 'font_size'])
    
    def get_span_tag(self,):
        span_scores = []
        span_num_occur = {}
        special = '[(_:/,#%\=@)]'

        for index, span_row in self.span_df.iterrows():
            score = round(span_row.font_size)
            text = span_row.text
            if not re.search(special, text):
                if span_row.is_bold:
                    score +=1 
                if span_row.is_upper:
                    score +=1
            span_scores.append(score)

        values, counts = np.unique(span_scores, return_counts=True)

        style_dict = {}
        for value, count in zip(values, counts):
            style_dict[value] = count
        #sorted(style_dict.items(), key=lambda x: x[1])

        p_size = max(style_dict, key=style_dict.get)
        idx = 0
        tag = {}
        for size in sorted(values, reverse = True):
            idx += 1
            if size == p_size:
                idx = 0
                tag[size] = 'p'
            if size > p_size:
                tag[size] = 'h{0}'.format(idx)
            if size < p_size:
                tag[size] = 's{0}'.format(idx)
        
        span_tags = [tag[score] for score in span_scores]
        self.span_df['tag'] = span_tags
        #span_df
    
    def get_heading_content(self,):
        headings_list = []
        text_list = []
        page_list = []
        tmp = []
        heading = ''                                                                                                                
        for index, span_row in self.span_df.iterrows():
            text = span_row.text
            tag = span_row.tag
            page_num = span_row.page_num
            header_not_found = True
            if 'h' in tag:
                headings_list.append(text)
                text_list.append('\n'.join(tmp))
                page_list.append(page_num)
                tmp = []
                heading = text
                header_not_found = False
            else:
                tmp.append(text)
        
        text_list.append('\n'.join(tmp))
        text_list = text_list[1:]
        text_df = pd.DataFrame(zip(page_list,headings_list, text_list),columns=['page_num','heading', 'content'])

        return text_df
    
    def text_cleansing(self,df):
        df['heading'] = df['heading'].apply(lambda x: x.replace("\n"," ").lower())
        df['content'] = df['content'].apply(lambda x: x.replace("\n"," ").lower())

        return df

    
    def get_headers_with_content(self,path,max_seq_length=150):
        return_dict = dict()
        try:
            if not self.check_if_file_extractable_or_not(path):
                raise Exception(f"file: {path} , not extractable.")
            
            self.read_pdf(path)
            self.get_block_dict()
            self.get_span_df()
            self.get_span_tag()
            df = self.get_heading_content()
            df = self.text_cleansing(df)

            for each_page in df['page_num'].unique():
                return_dict[each_page] = {}
                headings = df[df['page_num']==each_page]['heading'].to_list()
                headings = [i for i in headings if len(i.strip())>3]
                return_dict[each_page]['headers'] = headings
                return_dict[each_page]['content'] = df[df['page_num']==each_page]['content'].to_list()
        except Exception as err:
            print(err)
        
        return return_dict
    
    @staticmethod
    def get_index_slicer_range(size=0,window=100,overlap=0):
        prev = 0
        output = []
        for j in [i for i in range(window,size+window,window)]:
            output.append((prev,j))
            prev = j-overlap
        
        return output

    @staticmethod
    def get_sliced_text(text,max_seq_length):
        index_slicer = PdfHeaderGenerator.get_index_slicer_range(len(text),max_seq_length,5)
        #print(index_slicer)
        sliced_text = [text[i:j] for i,j in index_slicer]

        return sliced_text


if __name__ == '__main__':
    pdf_path = "C:/Users/achyuta.sahoo/Documents/work/NL_to_SQL/prod/nl_to_sql/data/pdf_docs/AMIND-Contingent Worker Agreement _ revised (1).pdf"
    header_gen = PdfHeaderGenerator()
    data = header_gen.get_headers_with_content(pdf_path)
    print(data)