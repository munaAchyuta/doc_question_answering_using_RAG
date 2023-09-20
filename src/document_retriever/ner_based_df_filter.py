import pandas as pd
import torch
torch.manual_seed(45)
from sentence_transformers import models as st_models
from sentence_transformers import SentenceTransformer, util
from torch import nn
from transformers import pipeline

import spacy
import concise_concepts
nlp = spacy.load(r'C:\Users\achyuta.sahoo\Downloads\en_core_web_lg-3.5.0\en_core_web_lg-3.5.0\en_core_web_lg\en_core_web_lg-3.5.0')

#from column_name_recognizer import ColumnNameRecognizer
from .column_name_recognizer import ColumnNameRecognizer

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
    #print(chunked_index)
	chunked_dfs = [df.loc[i,:].reset_index(drop=True) for i in chunked_index]

	return (chunked_index,chunked_dfs)

def get_fewshot_ner(custom_labels={},text=None):
    #custom_labels = {'order_date':['apple','pear','orange'],'vegetable':['brocoli','spinach','tomato'],'meat':['pork','beaf','lamb']}
    #text = """heat the oil in the large pan and add the onion, celery and carrots. add garlic and oregano."""
    _ = nlp.add_pipe("concise_concepts",config={"data":custom_labels})
    doc = nlp(text)
    out = doc.to_json()
    label_token = [{'label':i['label'],'token':text[i['start']:i['end']]} for i in out['ents']]

    return label_token

def filter_df_based_ner_spacy(df,text):
    custom_labels = get_df_custom_ner_data(df)
    filtered_cols = get_fewshot_ner(custom_labels,text)
    filtered_cols = [i['label'] for i in filtered_cols]
    return df.loc[:,filtered_cols]

def get_df_custom_ner_data(df):
    df_obj = df.select_dtypes(include=['object'])
    df_other_columns = list(set(df.columns)-set(df_obj.columns))
    df_other = df.loc[:,df_other_columns]
    labels = df_obj.to_dict('list')
    labels_oth = {i:[i] for i in df_other.columns}
    labels.update(labels_oth)

    return labels

def filter_df_with_ner_value(df,obj_columns,filters):
	for each_col in filters.keys():
		if each_col in obj_columns:
			val = filters[each_col]['column_variation']
			if val not in df.columns:
				df = df[df[each_col]==val]
	df = df.reset_index(drop=True)
	return df

def filter_df_based_ner_fuzzy(df,text):
    custom_labels = get_df_custom_ner_data(df)
    cer_obj = ColumnNameRecognizer()
    filtered_cols_dict = cer_obj.get_cer(text,custom_labels)
    filtered_cols = list(filtered_cols_dict.keys())
    obj_columns = list(df.select_dtypes(include=['object']).columns)
    df = filter_df_with_ner_value(df,obj_columns,filtered_cols_dict)
    
    return df.loc[:,filtered_cols]

def filter_df_based_ner(df,text,use_ner_model='fuzzy'):
	if use_ner_model is None:
		return df
	elif use_ner_model == 'spacy':
		return filter_df_based_ner_spacy(df,text)
	elif use_ner_model == 'fuzzy':
		return filter_df_based_ner_fuzzy(df,text)

if __name__ == '__main__':
	#d = {"id": [1, 2, 3, 4, 5],"sales": [9, 8, 7, 6, 5],"country": ["america", "india", "canada", "india", "englond"]}
	#df = pd.DataFrame(d)
	df = pd.read_excel(r'C:\Users\achyuta.sahoo\Downloads\sample_store_data_small.xlsx')
	df.drop(columns='Row ID',inplace=True)
	import pdb;pdb.set_trace()
	#indx,x = get_df_chunks(df)
	#print(f"length of chunked dfs: {len(x)}")
	#print("max length of chunked df: ",max([i.shape[0] for i in x]))
	#print(x[0])

	model = pipeline(task='table-question-answering',model='google/tapas-large-finetuned-wtq')
	question = "give me total sales in country united states of region west ?"
	df = filter_df_based_ner(df,question)
	df = df.astype(str)
	indx,x = get_df_chunks(df)

	#question = "what is the sales of country india ?"
	while True:
		for df_chunk in x:
			output = model(table=df_chunk, query=question)
			print(output)