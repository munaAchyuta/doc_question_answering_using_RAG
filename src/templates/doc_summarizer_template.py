template = '''
extract me answer from given document for provided question, if present return summary otherwise return 'No'.
NOTE: give attention to both word matching and context.

question: """{input_text}"""

document: """{document_text}"""
'''

template_2 = """are you sure of your answer? please recheck again. if answer present then return summary otherwise return 'No'"""

template_3 = '''
here is the document,

document: """{input_document}"""

question: "what is the document about ?"
'''