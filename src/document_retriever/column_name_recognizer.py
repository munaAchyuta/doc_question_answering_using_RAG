import copy
import re
from collections import defaultdict

import nltk
import numpy as np
#from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz, process

#from ..base.base import BaseClass

#nltk.download('wordnet')
#lemmatizer = WordNetLemmatizer()


class ColumnNameRecognizer():
    '''
    Class having methods for getting column names(mentioned config yaml file) from input text.
    '''

    def __init__(self) -> None:
        super().__init__()
        self.fuzzy_matching_threshold = 80#self.env_vars['cer']['fuzzy_matching_threshold']

    @staticmethod
    def levenshtein_distance(s1, s2):
        m = len(s1)
        n = len(s2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j],
                                       dp[i - 1][j - 1])
        return dp[m][n]

    @staticmethod
    def distance_filter(s1, s2):
        distance = ColumnNameRecognizer.levenshtein_distance(s1, s2)
        matching_score = 1 - distance / max(len(s1), len(s2))

        return matching_score

    def get_tokens(self, text):
        pattern = r'\s+'

        tokens = re.split(pattern, text)

        #tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def get_everygrams(self, tokens):
        n_tokens = nltk.everygrams(tokens, max_len=2)

        return list(n_tokens)

    def get_every_gram_string(self, text):
        tokens = self.get_tokens(text)
        every_grams = self.get_everygrams(tokens)

        n_tokens_strings = [
            " ".join(token_tuple) for token_tuple in every_grams
        ]

        return n_tokens_strings, every_grams

    def get_fuzzy_match(self, word_to_find, text):
        data_list, _ = self.get_every_gram_string(text)
        match = process.extractOne(word_to_find,
                                   data_list,
                                   scorer=fuzz.token_sort_ratio)
        if match[1] > self.fuzzy_matching_threshold:
            return (word_to_find, match[0], match[1])
        else:
            return (word_to_find, None, 0)

    def column_recognizer_fuzzy_match(self, columns, text):
        matching_list = []
        for each_column in columns:
            each_col_matches = []
            for each_column_variations in columns[each_column]:
                each_col_matches.append(
                    self.get_fuzzy_match(each_column_variations, text))
            matching_list.append(max(each_col_matches, key=lambda x: x[2]))
        matching_list = [i for i in matching_list if i[2] != 0]

        return matching_list

    def column_recognizer(self, text, columns=None):
        '''
        main method. with input args are text, columns dict[Optional]
        '''
        #self.logger.info(f"text: {text}")
        #self.logger.info(f"columns: {columns}")
        '''
        if columns is None:
            column_names = copy.deepcopy(
                self.db_mapping_info.get('schema_mapping', {}))
            column_names.pop(
                self.db_mapping_info.get('db_schema').get('table_name'))
            column_names_variations = self.db_mapping_info.get(
                'cer_entity_value_variations_mapping', {})

            # merge keys of entities and merge values
            columns = defaultdict(list)
            for col in column_names:
                columns[col].append(column_names[col])
            for col in column_names_variations:
                columns[col].extend(column_names_variations[col])
        else:
            column_names_variations = dict()
        '''
        cer = self.column_recognizer_fuzzy_match(columns, text)
        cer_dict = dict()
        for each_cer in cer:
            variation_match = False
            for each_col in columns:
                if each_cer[0] in columns[each_col]:
                    cer_dict[each_col] = {
                        'column_variation': each_cer[0],
                        'data_matched': each_cer[1],
                        'score': each_cer[2]
                    }
                    variation_match = True
            if not variation_match:
                cer_dict[each_cer[0]] = {
                    'column_variation': each_cer[0],
                    'data_matched': each_cer[1],
                    'score': each_cer[2]
                }

        #self.logger.info(f"column entity recogniser: {cer_dict}")

        return cer_dict
    
    def get_cer(self, text, columns={}):
        '''
        # get column names from given text
        '''
        # if columns entity recognition values not given.
        if len(columns) != 0:
            #use_model = {i: j for i, j in self.env_vars['cer']['use_model'].items() if j == 1}
            #if use_model.get('fuzzy_matching', None):
            if True:
                columns = self.column_recognizer(text,columns)
            else:
                columns = dict()

        return columns


if __name__ == '__main__':
    list_of_text = [
        "How does the patient count for each product vary across different states?",
        "Which disease area has the highest patient count overall?",
        "How has the patient count changed over time for each disease area?",
        "Are there any trends or patterns in the patient count data?",
        "How does the patient count for each product compare to the average for all products?",
        "Which disease areas have the highest patient counts compared to others?",
        "How does the patient count vary across different states for each disease area?",
        "Are there any significant differences in patient counts between different categories of products?",
        "How many patients were diagnosed with cancer in Texas in july 2021?",
        "What is the patient count for diabetes in Florida in Q3 2022?",
        "How did the number of patients with heart disease change in New York between 2019 and 2022?",
        "Can you provide the patient count for asthma in Illinois for July 2021?",
        "What is the trend for multiple sclerosis patients in California from Q1 2020 to Q4 2021?",
        "How does the patient count for HIV compare in Colorado between January and June of 2022?",
        "Has the patient count for arthritis increased or decreased in Arizona in the last 5 years?",
        "What is the current patient count for Parkinson's disease in Michigan for June 2022?",
        "How many Alzheimer's disease patients were there in Ohio in 2020?",
        "Compare the patient count for malaria in Nevada in July 2021 and July 2022."
    ]

    columns = {
        'PatientCount': 'patient_count',
        'Product': 'product_name',
        'State': 'location',
        'Deseasearea': 'disease_type',
        'Category': 'disease_class'
    }
    columns_derived = {
        'patient count': 'PatientCount',
        'patient': 'PatientCount',
        'desease area': 'Deseasearea',
        'desease': 'Deseasearea',
    }
    columns.update(columns_derived)

    #print(get_fuzzy_match('PatientCount',list_of_text[0]))

    for each_text in list_of_text:
        #print(f"input text:: {each_text}")
        #cer = ColumnNameRecognizer.column_recognizer_fuzzy_match(columns,each_text)
        #cer = {columns_derived.get(i[0],i[0]):i[1] for i in cer}
        #print(f"columns found before:: {cer}")
        # distance filter using lavenstein distance
        #cer = {i:j for i,j in cer.items() if ColumnNameRecognizer.distance_filter(i,j)>=0.5}
        #print(f"columns found:: {cer}")
        ColumnNameRecognizer().column_recognizer(each_text, columns)
        print("\n")
