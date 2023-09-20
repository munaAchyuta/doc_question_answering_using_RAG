import os
import platform
import pdf2image
import pytesseract
from pytesseract import Output, TesseractError
from img2table.document import PDF
from img2table.ocr import TesseractOCR


if platform.system() == 'Windows':
    # windows download-link :- https://digi.bib.uni-mannheim.de/tesseract/
    # linux/ubuntu installation link :- https://tesseract-ocr.github.io/tessdoc/Installation.html
    # download tessract-ocr exe file. extract files to selected path. give path here.
    # OR download tessract-ocr exe file. give path here. it'll extract files in selected path. then use that path here.
    #tesseract_exe_path = r"./applications"
    #tesseract_exe = r"C:\Users\achyuta.sahoo\Documents\work\NL_to_SQL\prod\document_retriever_bkp\document_retriever\applications\tesseract.exe"
    tesseract_exe = r"./applications/tesseract.exe"
    tesseract_exe = os.path.abspath(tesseract_exe)
    tesseract_exe_path,tail = os.path.split(tesseract_exe)
    print(tesseract_exe)
    print(tesseract_exe_path)
    pytesseract.pytesseract.tesseract_cmd = tesseract_exe


def get_text_from_image(path):
    images = pdf2image.convert_from_path(path)
    data_pages = []
    for i in range(len(images)):
        pil_im = images[i] # assuming that we're interested in the first page only
        ocr_dict = pytesseract.image_to_data(pil_im, lang='eng', output_type=Output.DICT)
        # ocr_dict now holds all the OCR info including text and location on the image
        text = " ".join(ocr_dict['text'])
        #print(text)
        data_pages.append({"file_name":path,
                           "page_number":i,
                           "page_content":text})
    
    return data_pages

def get_table_from_image(path):
    # Instantiation of the pdf
    pdf = PDF(src=path)
    #print(help(pdf.extract_tables()))

    # Instantiation of the OCR, Tesseract, which requires prior installation
    ocr = TesseractOCR(lang="eng",tessdata_dir=tesseract_exe_path)
    
    # Table identification and extraction
    pdf_tables = pdf.extract_tables(ocr=ocr)
    print(pdf_tables)

    # We can also create an excel file with the tables
    #pdf.to_xlsx('tables.xlsx',ocr=ocr)
    


#'''
#import easyocr
#reader = easyocr.Reader(['en'])
#result = reader.readtext('test.jpg', detail = 0)
#print(result)

if __name__ == "__main__":
    pdf_path = r"C:\Users\achyuta.sahoo\Documents\work\NL_to_SQL\prod\document_retriever_bkp\document_retriever\data\pdf_docs\CW9930-Amendment 01 to SOW 11 - Evalueserve.pdf"
    data_pages = get_text_from_image(pdf_path)
    #data_pages = get_table_from_image(pdf_path)
    for i in data_pages:
        print(i)
        print("\n====\n")