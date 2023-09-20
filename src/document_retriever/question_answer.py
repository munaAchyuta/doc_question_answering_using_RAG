from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd

from ..base.base import BaseClass
from ..db_opts.model_loader import ModelLoader
from .ner_based_df_filter import filter_df_based_ner


class QuestionAnsweringModel(BaseClass):
    def __init__(self,qa_model_name=None,tqa_model_name=None,similarity_model=None) -> None:
        super().__init__()
        self.cer_for_df = self.env_vars['cer']
        self.cer_for_df_use_it = self.cer_for_df['use_it']
        if self.cer_for_df_use_it:
            self.use_cer_model_tmp = [i for i,j in self.cer_for_df['use_model'].items() if j==1]
            self.use_cer_model = None if len(self.use_cer_model_tmp)==0 else self.use_cer_model_tmp[0]
        
        self.use_sort_of_sent_from_text = self.doc_retrieval_config['document_retrieval_structure']['return_docs_using_question_answer']['use_sort_of_sent_from_text']
        self.use_sort_of_sent_from_text_useit = self.use_sort_of_sent_from_text['use_it']
        self.use_sort_of_sent_from_text_chunk_size = self.use_sort_of_sent_from_text['chunk_size']
        
        model_loader = ModelLoader()
        # q&a
        if qa_model_name is None:
            self.qa_model = model_loader.get_qa_model(task_name='question-answering',model_name='distilbert-base-cased-distilled-squad')
        else:
            self.qa_model = model_loader.get_qa_model(task_name='question-answering',model_name=qa_model_name)

        # table q&a
        if tqa_model_name is None:
            self.tqa_model = model_loader.get_qa_model(task_name='table-question-answering',model_name='google/tapas-large-finetuned-wtq')
        else:
            self.tqa_model = model_loader.get_qa_model(task_name='table-question-answering',model_name=tqa_model_name)
        
        # similarity model
        if similarity_model is None:
            self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.similarity_model = model_loader.get_encoder(model_name=similarity_model)
    
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
        index_slicer = QuestionAnsweringModel.get_index_slicer_range(len(text),max_seq_length,overlap)
        #print(index_slicer)
        sliced_text = [text[i:j] for i,j in index_slicer]

        return sliced_text
    
    def filter_rows_from_context(self,question,context,max_seq_length=384):
        '''
        filter rows from table using cosine similarity.
        purpose: if tqa-model fails with bigger size of table, then using cosine similarity,
        order top-to-bottom. and then take top N rows.
        '''
        sliced_texts = QuestionAnsweringModel.get_sliced_text(context,max_seq_length,overlap=0)
        text_with_score = []
        for index,each_text in enumerate(sliced_texts):
            text_with_score.append((index,self.similarity_score(question,each_text),each_text))
        text_with_score = sorted(text_with_score,key=lambda x: x[1],reverse=True)

        return text_with_score

    def get_answer_given_question_context(self,question,context,max_seq_length=384):
        '''
        input: question & context
        output: dictionary having answer.
        '''
        context = context.replace('\n',' ')
        context = ' '.join(context.split())
        context = context.lower()

        if self.use_sort_of_sent_from_text_useit:
            text_with_score = self.filter_rows_from_context(question,context,self.use_sort_of_sent_from_text_chunk_size)
            self.logger.info(f"get_answer_given_question_context --> chunked text with score: {text_with_score}")
            sorted_context = " ".join([i[2] for i in text_with_score])[:max_seq_length]
            self.logger.info(f"sorted_context: {sorted_context}")
            output = self.qa_model(question, sorted_context)
        else:
            output = self.qa_model(question, context)

        self.logger.info(f"get_answer_given_question_context output: {output}")

        return output

    def get_answer_given_question_table(self,question,table):
        '''
        input: question & context
        output: dictionary having answer.
        LIMITATION:
            Ref: https://github.com/google-research/tapas/issues/14
        Approach to solve problem of table-qa.
            1. use zero/few shot NER against table columns as lables and input question
            2. then apply filter to shorten table.
        '''
        output = ""
        try:
            self.logger.info(f"get_answer_given_question_table table: \n{table}")
            #table.fillna('NA',inplace=True)
            table_bkp = table.copy()

            # format table
            table = table.astype(str)
            
            # rectify table
            table = self.rectify_df(table)

            self.logger.info(f"get_answer_given_question_table rectify_df table: \n{table}")

            if self.cer_for_df_use_it or (table_bkp.shape[0]==table.shape[0]):
                self.logger.info(f"going ner_based_df_filter get_answer_given_question_table.")
                if self.cer_for_df_use_it:
                    table = filter_df_based_ner(table_bkp,question,use_ner_model=self.use_cer_model)

                # format table
                table = table.astype(str)
                
                # rectify table
                table = self.rectify_df(table)

                output = self.tqa_model(table=table, query=question)
                self.logger.info(f"get_answer_given_question_table output: {output}")
                output = output['cells']
            else:
                self.logger.info(f"going default get_answer_given_question_table.")
                # format table
                #table = table.astype(str)
                
                # rectify table
                #table = self.rectify_df(table)

                output = self.tqa_model(table=table, query=question)
                self.logger.info(f"get_answer_given_question_table output: {output}")
                output = output['cells']
            '''
            # chunked table
            chunked_tables = self.get_df_chunks(table)

            output = []
            for table_chunk in chunked_tables:
                chunk_output = self.tqa_model(table=table_chunk, query=question)
                self.logger.info(f"get_answer_given_question_table output: {chunk_output}")
                chunk_output = chunk_output['cells']
                output.append(chunk_output)
            '''
        except IndexError as ierr:
            self.logger.error(f"get_answer_given_question_table error: {ierr}")
            self.logger.info(f"trying with qa-model::::::::")
            try:
                context = self.get_df_to_text(question,table)
                output = self.get_answer_given_question_context(question,context)
                output = output['answer']
                self.logger.info(f"get_answer_given_question_table with qa-model -> output: {output}")
            except Exception as err:
                self.logger.error(err)
                output = ""
        except TypeError as terr:
            self.logger.error(terr)
        except Exception as err:
            self.logger.error(err)
        
        return output
    
    def get_indexes_of_cols_and_rows(self,X):
        def get_column_rows(X):
            fst_not_found = True
            max_ind = 0
            for ind,val in enumerate(X):
                if val == 'nan' and fst_not_found:
                    continue
                fst_not_found = False
                max_ind = ind
                break
            return list(range(0,max_ind))

        def get_rows(X):
            all_seq = []
            tmp_seq = []
            prev = ""
            for ind,val in enumerate(X):
                if val != 'nan':
                    if prev == 'nan':
                        all_seq.append(tmp_seq)
                        tmp_seq = []
                    tmp_seq.append(ind)
                else:
                    tmp_seq.append(ind)
                prev = val
            all_seq.append(tmp_seq)    
            return all_seq
        
        Y = []
        col_rows = get_column_rows(X)
        remaining_inds = [i for i in range(0,len(X)) if i not in col_rows]
        X_new = list(map(X.__getitem__, remaining_inds))
        remaining_rows = get_rows(X_new)
        remaining_rows = [list(map(lambda x:x+len(col_rows), j)) for j in remaining_rows]
        Y = [col_rows] + remaining_rows

        self.logger.info(f"get_indexes_of_cols_and_rows -> Y: {Y}")
        
        return Y
    
    def rectify_df(self,df):
        '''
        create right structure dataframe.
        purpose: if table is not in proper format, then make it.
        NOTE: this is one scenario. like this user has to handle many other possible scenarios.
        '''
        if 'unnamed' in df.columns[0].lower():
            fst_col_recs = df.iloc[:,0].tolist()
            
            Y = self.get_indexes_of_cols_and_rows(fst_col_recs)
            
            cols = df.columns
            new_cols = []
            new_rows = []
            for ind,j in enumerate(Y):
                if ind == 0:
                    for col_ind in range(0,len(cols)):
                        new_col_str = ' '.join(df.iloc[j,col_ind].values)
                        new_col_str = new_col_str.replace('nan','')
                        new_cols.append(new_col_str)
                else:
                    tmp_row = []
                    for col_ind in range(0,len(cols)):
                        tmp_row_str = ' '.join(df.iloc[j,col_ind].values)
                        tmp_row_str = tmp_row_str.replace('nan','')
                        tmp_row.append(tmp_row_str)
                    new_rows.append(tmp_row)
            
            df = pd.DataFrame(new_rows, columns=new_cols)
        
        self.logger.info(f"rectify_df -> structured df size: {len(df)}")
        
        return df
    
    def filter_rows_from_df(self,question,df):
        '''
        filter rows from table using cosine similarity.
        purpose: if tqa-model fails with bigger size of table, then using cosine similarity,
        order top-to-bottom. and then take top N rows.
        '''
        cols = df.columns
        row_with_score = []
        for index,row in df.iterrows():
            row_text = ""
            for col in cols:
                row_text += f"{col}: {row[col]}"
            row_with_score.append((index,self.similarity_score(question,row_text)))
        row_with_score = sorted(row_with_score,key=lambda x: x[1],reverse=True)

        row_with_score_indexes = [i[0] for i in row_with_score]
        df_new = df.iloc[row_with_score_indexes]

        return df_new
    
    def get_df_chunks(df,max_sequence_len=512,considered_len=480):
        '''
        max_sequence_len = 10
        considered_len = 5
        '''
        col_str = " ".join(df.columns)
        chunked_index = []
        index = 0
        start_indx = index
        while True:
            #print(f"start_indx: {start_indx} , index: {index}")
            if index > df.shape[0]:
                break

            if len(col_str + " ".join(df.loc[start_indx:index, :].values.flatten().tolist())) < considered_len:
                index += 1
                continue
            else:
                if index > 0:
                    if start_indx == index:
                        index += 1
                elif len(list(range(start_indx,index-1))) == 0:
                    index = df.shape[0]+1
                    break
                else:
                    pass
                
                if start_indx >= index-1:
                    chunked_index.append(list(range(start_indx,start_indx+1)))
                else:
                    chunked_index.append(list(range(start_indx,index-1)))
                start_indx = index
        
        chunked_index.append(list(range(start_indx,index-1)))
        print(chunked_index)
        chunked_dfs = [df.loc[i,:].reset_index(drop=True) for i in chunked_index]

        return chunked_dfs
    
    def get_df_to_text(self,question,df):
        '''
        convert to table to text.
        purpose: if tqa-model fails, then use qa-model. 
                for using qa-model, convert table to text-context.
        '''
        new_df = self.filter_rows_from_df(question,df)

        cols = new_df.columns
        row_texts = []
        for index,row in new_df.iterrows():
            row_text = ""
            for col in cols:
                row_text += f"{col}: {row[col]}"
            row_texts.append(row_text)
        
        return "\n".join(row_texts)
    
    def similarity_score(self,text1,text2):
        '''
        given two texts,
        get vectors using sentence_transformer. and then using cosine-metric get similarity score.
        '''
        #Compute embedding for both lists
        embedding_1= self.similarity_model.encode(text1, convert_to_tensor=True)
        embedding_2 = self.similarity_model.encode(text2, convert_to_tensor=True)

        score = util.pytorch_cos_sim(embedding_1, embedding_2)
        return score[0][0]
    
    def get_fewshot_ner(self,custom_labels={},text=None):
        import spacy
        import concise_concepts
        nlp = spacy.load(r'C:\Users\achyuta.sahoo\Downloads\en_core_web_lg-3.5.0\en_core_web_lg-3.5.0\en_core_web_lg\en_core_web_lg-3.5.0')
        #custom_labels = {'fruit':['apple','pear','orange'],'vegetable':['brocoli','spinach','tomato'],'meat':['pork','beaf','lamb']}
        #text = """heat the oil in the large pan and add the onion, celery and carrots. add garlic and oregano."""
        _ = nlp.add_pipe("concise_concepts",config={"data":custom_labels})
        doc = nlp(text)
        out = doc.to_json()
        label_token = [{'label':i['label'],'token':text[i['start']:i['end']]} for i in out['ents']]

        return label_token
    
    def filter_df_based_ner(self,df,text):
        custom_labels = self.get_df_custom_ner_data(df)
        filtered_cols = self.get_fewshot_ner(custom_labels,text=text)
        filtered_cols = [i['label'] for i in filtered_cols]

        return df.loc[:,filtered_cols]

    def get_df_custom_ner_data(self,df):
        df_obj = df.select_dtypes(include=['object'])
        df_other_columns = list(set(df.columns)-set(df_obj.columns))
        df_other = df.loc[:,df_other_columns]
        labels = df_obj.to_dict('list')
        labels_oth = {i:[i] for i in df_other.columns}
        labels.update(labels_oth)

        return labels



if __name__=="__main__":
    qamodel = QuestionAnsweringModel()
    question = "Where do I live?"
    context = "My name is Merve and I live in İstanbul."
    print(qamodel.get_answer_given_question_context(question,context))

    # prepare table + question
    data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
    table = pd.DataFrame.from_dict(data)
    question = "how many movies does Leonardo Di Caprio have?"
    print(qamodel.get_answer_given_question_table(question,table))