import io
import json
import os
import sqlite3
import pandas as pd

from .base import BaseClass


class SqliteDbOpts(BaseClass):
    db_conn = {}
    def __init__(self,feedback_loop_path_sqlite=None,feedback_loop_path_sqlite_table=None) -> None:
        super().__init__()
        if feedback_loop_path_sqlite is not None:
            self.feedback_loop_path_sqlite = feedback_loop_path_sqlite
        if feedback_loop_path_sqlite_table is not None:
            self.feedback_loop_path_sqlite_table = feedback_loop_path_sqlite_table
        
        self.create_db()
        self.create_table()
    
    def create_db(self,):
        if not SqliteDbOpts.db_conn.get(self.feedback_loop_path_sqlite,None):
            SqliteDbOpts.db_conn[self.feedback_loop_path_sqlite] = sqlite3.connect(self.feedback_loop_path_sqlite, check_same_thread=False)
        self.conn = SqliteDbOpts.db_conn[self.feedback_loop_path_sqlite]

    def create_table(self,):
        try:
            self.conn.execute(
                f'''CREATE TABLE {self.feedback_loop_path_sqlite_table} 
                (Id INTEGER PRIMARY KEY AUTOINCREMENT, 
                Date DATETIME NOT NULL DEFAULT (datetime('now')), 
                Api TEXT,
                Question TEXT,
                Answer TEXT, 
                Other TEXT,
                Flag REAL);'''
                )
        except Exception as err:
            print(err)

    def insert_record(self,data={'api':'document_retriever','question':'NA','answer':'NA','other':'NA','flag':0}):
        sqlite_insert_with_param = f"""INSERT INTO {self.feedback_loop_path_sqlite_table}
                        (Api,Question,Answer,Other,Flag) 
                        VALUES (?, ?, ?, ?, ?);"""

        data_tuple = (data['api'], data['question'], data['answer'], f"{data['other']}", data['flag'])
        self.conn.execute(sqlite_insert_with_param, data_tuple)
        self.conn.commit()

    def update_record(self,data={}):
        pass

    def delete_record(self,data={}):
        pass

    def get_dict(self,list_of_data):
        '''
        input: list of tuples
        convert list of tuples to list of dict. use column names for dict keys.
        output: list of dict
        '''
        columns = ('id','date','api','question','answer','other','flag')
        dict_data = [dict(zip(columns,data)) for data in list_of_data]

        return dict_data

    def get_records(self,where_clause=None):
        if where_clause is None:
            cursor = self.conn.execute(f"SELECT * FROM {self.feedback_loop_path_sqlite_table};")
        else:
            cursor = self.conn.execute(f"SELECT * FROM {self.feedback_loop_path_sqlite_table} WHERE {where_clause};")
        list_of_tuples = [(i[0],i[1],i[2],i[3],i[4],eval(i[5]),i[6]) for i in cursor]

        list_of_dict = self.get_dict(list_of_tuples)

        return list_of_dict
    
    def add_feedback_loop_data(self,data):
        self.insert_record(data)

class JsonFileDrProcessedFilesOpts(BaseClass):
    def __init__(self) -> None:
        super().__init__()
        self.processed_file_path = self.doc_retrieval_config['processed_file_path']
        self.processed_error_file_path = self.doc_retrieval_config['processed_error_file_path']
    
    def create_file(self,):
        if os.path.isfile(self.processed_file_path) and os.access(self.processed_file_path, os.R_OK):
            # checks if file exists
            print("File exists and is readable")
        else:
            print("Either file is missing or is not readable, creating file...")
            with io.open(self.processed_file_path, 'w') as db_file:
                db_file.write(json.dumps({"index_count": 0, "data": {}}))
    
    def create_error_file(self,):
        if os.path.isfile(self.processed_error_file_path) and os.access(self.processed_error_file_path, os.R_OK):
            # checks if file exists
            print("File exists and is readable")
        else:
            print("Either file is missing or is not readable, creating file...")
            with io.open(self.processed_error_file_path, 'w') as db_file:
                db_file.write(json.dumps({"index_count": 0, "data": {}}))


    def read_data(self,):
        with open(self.processed_file_path, "r") as processed_obj:
            processed_data = json.loads(processed_obj.read())

        return processed_data
    
    def read_error_data(self,):
        with open(self.processed_error_file_path, "r") as processed_obj:
            processed_data = json.loads(processed_obj.read())

        return processed_data


    def add_data(self,data):
        processed_data = self.read_data()
        max_index = processed_data["index_count"]
        data.update({"index": max_index + 1})
        processed_data["data"].update(data)
        processed_data["index_count"] = max_index + len(data)

        with open(self.processed_file_path, "w") as processed_obj:
            processed_obj.write(json.dumps(processed_data))

        return None
    
    def add_error_data(self,data):
        processed_data = self.read_error_data()
        max_index = processed_data["index_count"]
        data.update({"index": max_index + 1})
        processed_data["data"].update(data)
        processed_data["index_count"] = max_index + len(data)

        with open(self.processed_error_file_path, "w") as processed_obj:
            processed_obj.write(json.dumps(processed_data))

        return None
    
    def get_records(self,):
        processed_data = self.read_data()
        return processed_data

class JsonFileDrProcessedSqliteDataOpts(BaseClass):
    def __init__(self,file_path=None) -> None:
        super().__init__()
        if file_path is None:
            self.processed_feedback_qa_records_file_path = self.doc_retrieval_config['processed_feedback_qa_records_file_path']
        else:
            self.processed_feedback_qa_records_file_path = file_path
    
    def create_file(self,):
        if os.path.isfile(self.processed_feedback_qa_records_file_path) and os.access(self.processed_feedback_qa_records_file_path, os.R_OK):
            # checks if file exists
            print("File exists and is readable")
        else:
            print("Either file is missing or is not readable, creating file...")
            with io.open(self.processed_feedback_qa_records_file_path, 'w') as db_file:
                db_file.write(json.dumps({"index_count": 0, "data": {}}))

    def read_data(self,):
        with open(self.processed_feedback_qa_records_file_path, "r") as processed_obj:
            processed_data = json.loads(processed_obj.read())

        return processed_data


    def add_data(self,data):
        processed_data = self.read_data()
        processed_data.update(data)

        with open(self.processed_feedback_qa_records_file_path, "w") as processed_obj:
            processed_obj.write(json.dumps(processed_data))

        return None
    
    def get_records(self,):
        processed_data = self.read_data()
        return processed_data


JsonFileDrProcessedFilesOpts().create_file()

#JsonFileOpts().create_nl_sql_feedback()