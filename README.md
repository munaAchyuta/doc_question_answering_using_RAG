# Document Retrieval: Process pdf documents and retrieve documents given user query.
This is a very simple application with minimal features used for document processing to create knowledge base and then given a query looking at knowledge base gives top match records. to handle complex functionality, you are welcome to contribute.

## Example Flow-Diagram
<img src="example-FLowDiagram-of-DocumentRetrieval.avif" width="728"/>

## Pre-requisites:
  1. use `requiremnets_doc_retrieval.txt` to install python packages.
      NOTE: if python incompatibility issue arise, then best create separate python virtual environment.
  2. to convert pdf-images to text, we need to install tesseract-OCR. please follow below guidelines. if already file present in `./applications` path then skip below steps.
      * windows download-link :- https://digi.bib.uni-mannheim.de/tesseract/
      * linux/ubuntu installation link :- https://tesseract-ocr.github.io/tessdoc/Installation.html
      * WIndows:
        - download tessract-ocr exe file. extract files to selected(`./applications`) path.
        - OR download tessract-ocr exe file. explicitly give path `./applications`. it'll extract files in selected(`./applications`) path.
      * NOTE: after install of pytesseract on windows, during use, if issue comes related to poppler, then 
      install `conda install -c conda-forge poppler`. if issue still persist, then please google to resolve.
  3. under `./config` directory, two config files present.

      a. update `config.yaml` for application level and doc_retrieval_config configuration.
  
  4. execute `python -m src.document_processor.processer`. this will process pdf files and uploads to vector DB.
  5. modify or add template in `doc_summarizer_template.py`. this one used for summary generation prompt in openAI. currently `template` is used.


## Configuration Details
configuration regarding Document Retrieval APP present under `doc_retrieval_config` section of `./config/config.yaml`.
here are short description of each config variables.

* `document_path` :- path having pdf files
* `vector_db_path` :- path to vector DB
* `use_db` :- this one holds configuration regarding which DB will be used for document retrieval.
* `nlp_preprocessing` :- this one hold parameter regarding feature generation of document processing.
    1. `text_chunk_config` :- configuration regarding text chunk will be used for upload/retrieval in vector db.
    1. `topics` :- generate topics from text. which later used in creating topic embeddings and then for retrieval.
    2. `summary` :- generates summary/overview/abstract from text.
* `use_vector_embedding` :- this config variable holds information about embedding model and it's configurations.
    1. `use_it` :- true/false . if this is true, then it'll be used otherwise not.
    2. `model` :- name of model to be used to generate vectors
    3. `use_lower_embedding_size` :- used to lower the vector size.
    4. `max_seq_length` :- maximum sequesnce length of text used to feed to model
    5. `embedding_size` :- model's default vector size
    6. `use_local_model` :- local model config, which has use_it and model(path).
* `document_upload_structure` :- this one holds document upload configuration.
  * `use_page_content_similarity` :- this variable used for creating content(raw text) vectors. it has it's config whether to use or not and table/collection name where vectors to be stored.
  * `use_topic_similarity` :- this variable used for creating topic vectors. it has it's config whether to use or not and table/collection name where vectors to be stored.
  * `use_both` :- (RECOMMENDED)this variable used for creating both content(raw text) and topic vectors. it has it's config whether to use or not and table/collection name where vectors to be stored.
* `feedback_loop_path_dr_sqlite` :- path to creating sqlite db for feeding document retrieval input and output.
* `feedback_loop_path_dr_sqlite_table` :- table/collection name where data will be stored.
* `processed_file_path` :- path to json file. this one stores file name which got processed successfully.
* `processed_error_file_path` :- path to json file. this one stores file name which failed to processed.
* `processed_feedback_qa_records_file_path` :- path to json file. this one stores max-record id from sqlite-db(doc-retrieval feedback data) which got added to qdrant-db for finding duplicate question. so, next time when processor runs it will not add already same data again.
* `document_retrieval_structure` :- this one holds retrieval configuration.
  * `return_similar_docs` :- this variable holds configuration about matching documents. DEFAULT: if all false, then text/content matching used.
      1. `use_it` :- true/false . if this is true, then it'll be used otherwise not.
      2. `use_of_retrieval_order` :- config about how document retrieval to be done.

          a. `header_to_text` :- this one says use headers matching first and then text/content matching. #TOBE-DONE based on project need.
          
          b. `header_topic_to_text` :- this one says use headers & topics matching first and then text/content matching. #TOBE-DONE based on project need.
  * `return_summary` :- this variable holds configuration about matching documents along with returning summary using OpenAI.  DEFAULT: if all false, then text/content matching used.
      1. get top-N records/documents matched from `use_page_content_similarity` DB.
      2. then use `use_topic_similarity` DB, to get raw-page content.
      3. finally use summary API to get summary.
  * `return_docs_using_question_answer` :- this variable holds configuration about matching documents along with returning exact either factual/descriptive answer using Question-Answering model and Table-Question-Answering model. DEFAULT: if all false, then text/content matching used.
      1. get top-N records/documents matched from `use_page_content_similarity` DB.
      2. then use `use_topic_similarity` DB, to get raw-page content.
      3. finally use question-answering model API to extract factual/descriptive answer.
  * `use_reranking` :- used for re-ranking for documents after retrieval from vector db.
* `retrieval_docs_limit` :- number of documents should be returned. 
* `retrieval_docs_window` :- number of neighouring documents to add to matched document before returning response.
* `duplicate_question_finder` :- this will be used for duplicate question finding.


## Start Service:
* on this path `(where README file resides.)` , Once execution of `document_processor.processer.py` completed. then execute `uvicorn src.doc_retrieval:app --port 8001` to start document retrieval service.

* `NOTE:` It is mandatory that execution of `document_processor.processer.py` and `uvicorn src.doc_retrieval:app --port 8001` should be done sequentially with this order. Reason is these two process can't be run paralelly as it throws error w.r.t qdrant thread issue. while application is in running state, if new documents arrives, then user should take out some time 1.shutdown document_retrieval service and 2.process new pdf documents if arrived and then start document_retrieval service again.
* `IMPORTANT Point on Scalability`: if data size huge and performance degrades, then it is suggested to try out `production grade usage of qdrant application` and trying out better usage of `vector embeddings` like lowering it's dimensions.
* `IMPORTANT point on Retrieval Performance`: if document retrieval w.r.t query is not good, then here are few points, user should look into.
    1. data quality which is getting feed to vector DB. user should figure out what is right quality of data w.r.t it's project. and based on that should add data-quality functions in module.
    2. deciding better vector-embedding model. again here also user should play with differnet model to figure out best model for it's data. and also if required user should fine-tune model and use that for getting vectors.
    3. using hybrid approach(Text-Ranking & Vector-Ranking)
    4. finally always look for help from qdrant documentations, sentence-transformers documentations and Google.


## Usage:

### retrive_document
#### Input:

```
curl -X 'PUT' \
  'http://127.0.0.1:8001/retrive_document/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input_text": "give me former employer details from CONTINGENT WORKER CONFIDENTIALITY AGREEMENT ?",
  "other": "string"
}'
```
#### Output:
```
{
  "output": [
    {
      "file": "./data/pdf_docs\\AMIND-Contingent Worker Agreement _ revised (1).pdf",
      "page": 4,
      "seq_ids": [
        5,
        6,
        7,
        8,
        9
      ],
      "answer": "  employment agreement between contingent worker and applied.     5.   ownership of documents   all  data,  including  drawings,  specifications,  designs  and  other  information  furnished  by  applied  t o  contingent worker in connection with his/her work a ork assignment will remain the sole and exclusive property  of applied.   the contingent worker shall return these items  to applied upon termination of his/her work  assignment or at any time upon request by applied.   6.   information of former employer   con   contingent worker shall not disclose or use for applied’s benefit any confidential or proprietary information  of contingent worker’s former employer(s) or any  other third parties to  which contingent worker has  a  confidentiality obligation. the  contingen ingent worker must  not bring onto the premises  of applied any  non- public documents or any other property belonging to any former employer(s) or any other third parties unless  permitted to do so in advance in writing by such party.   7.   status of continge tingent worker   contingent worker acknowledges that he/she is an employee of service provider, not applied.  in no event  will contingent worker be eligible for or entitled to any benefits under any of applied’s employee benefit  plans,  arrangements, and  pol"
    },
    .....
  ]
}
```
### generate answer given question & table
#### input
```
curl -X 'POST' \
  'http://127.0.0.1:8001/get_answer_given_question_table/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "which language chris use ?",
  "table": [{"Name": "Aditya", "Roll": 1, "Language": "Python"},{"Name": "Sam", "Roll": 2, "Language": "Java"},{"Name": "Chris", "Roll": 3, "Language": "C++"},{"Name": "Joel", "Roll": 4, "Language": "TypeScript"}]
}'
```
#### output
```
"C++"
```
### generate answer given question & context
#### input
```
curl -X 'POST' \
  'http://127.0.0.1:8001/get_answer_given_question_context/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "is contigent worker allowed take participate in awards ceremony at AMAT ?",
  "context": " \n\nnetworking, computing, and other electronic communication or data storage systems  (collectively, the \n\"Applied  Communication  and  Data  Systems\").  Because  the  Applied  Communication  and  Data \nSystems  are  owned  and  controlled  by  Applied  and  intended  primarily for  business  use,  Contingent \nWorker should have no expectation of personal privacy with respect to the Applied Communication and \nData Systems, including the information that they access, store, or transmit through the systems.    \n \n\nb.  Applied can access,  monitor, record, or search any electronic resources, information technology assets, \nor  workspace, including the  contents  of any  files  or  information maintained or passed  through these \nsources, at any time for any reason or no reason, with or without notice.   \n\n \n\nc.  Contingent Worker acknowledges and grants consent to Applied to collect, hold and process his/her personal \ndata  in accordance  with applicable  law for such  purposes  necessary  for the  continued engagement of the \nContingent  Worker  with  Applied.  Where  consent  is  required  for  any  processing  of  personal  data,  the \nContingent Worker agrees that  Applied may collect, store, and  process  personal data  (including sensitive \npersonal data) provided by him/her (and where appropriate by third parties inside or outside the U.S.) for the \naforesaid  purpose, including  the  release  and  transfer of  such  information to  third parties  (including any \naffiliated entities of Applied), whether or not in other jurisdictions.  The Contingent Worker shall have the \nright  to  review,  amend,  delete  or  withdraw  consent  for  the  aforementioned  collection,  handling  and \nprocessing of his/her personal data by Applied. In the event the Contingent Worker withdraws such consent, \nApplied shall have the right to terminate the engagement with such Contingent Worker. \n\n \n10. \n\nACTIVITIES AND AWARDS \n\nContingent Worker acknowledges that Contingent Workers are not eligible to participate in any company-\nsponsored  activity of a  social  or business  nature that takes  place within/outside of regular work hours  or \ninside/outside  Applied premises.  Additionally, Contingent Workers  are  not  eligible  to  participate in  any \nApplied sponsored award or recognition programs. \n\n11. \n\nINDEMNIFICATION \n\na.  If Contingent Worker uses Confidential Information beyond the scope permitted under this Agreement, \nor"
}'
```
#### output
```
"Contingent Workers  are  not  eligible"
```

## Duplicate query Finding:

* a simple solution for finding similar query from past records. approache used here is.. 
  1. text-matching(given vectors of input query and past records, use cosine similarity metrics to get most similar records from past.). using qdrant-db vector search, problem solved.
* configuration details are present in `config.yaml`. `duplicate_question_finder` needs to be true in order to use.
* `Execution Flow:` 
  1. execute `python -m src.document_processor.qna_processor` to upload question and answer from sqlite DB. records went to this DB using API endpoint `/add_feedback_loop_data/`.
  2. then if `duplicate_question_finder` section made true, then during document_retireval , first duplicate query finding will be used.


## REFERENCES
* https://stackoverflow.com/questions/63461262/bert-sentence-embeddings-from-transformers .  for fine-tuning embedding models on custom data.
* https://huggingface.co/docs/transformers/tasks/question_answering . for fine-tuning question-answering embedding models on custom data.
* https://huggingface.co/docs/transformers/model_doc/tapas . for fine-tuning table question-answering embedding models on custom data.
* https://qdrant.tech/articles/hybrid-search/ . qdrant search improvement.
* https://www.sbert.net/docs/training/overview.html . getting sentence vectors locally.

# Features

* TODO

# Credits
