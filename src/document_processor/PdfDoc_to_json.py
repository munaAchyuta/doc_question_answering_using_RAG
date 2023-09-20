from lxml import html
import re
import json
from bs4 import BeautifulSoup
from docx2python import docx2python
import mammoth
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from tika import parser
from pdfminer.high_level import extract_text


def get_html_from_docx(file_name):
    '''
    input: file-name
    output: html data
    '''
    f = open(file_name, 'rb')
    docu = mammoth.convert_to_html(f)
    f.close()

    return docu.value.encode('utf-8')


def get_html_from_pdf(file_name):
    '''
    input: file-name
    output: html data
    '''
    output_string = StringIO()
    with open(file_name, 'rb') as fin:
        extract_text_to_fp(fin, output_string, laparams=LAParams(), output_type='html', codec=None)

    return output_string.getvalue()


def parse_html(file_name=None, html_data='', pattern_to_look_for=".//strong"):
    '''
    input: file-name & pattern to look for
    '''
    if file_name is not None:
        with open(file_name, 'r', encoding='utf-8') as f:
            html_data = f.read()
    
    out = {}
    if pattern_to_look_for==".//strong":
        x = html.fromstring(html_data)
        # find bold tags by class name
        for b in x.xpath(pattern_to_look_for):
            # get bold text
            if b.text is None:
                continue
            
            # get text between current bold up to next br tag.
            out[b.text] = ''.join(b.xpath("./following::text()[1]"))
    
    return out

def get_json_from_docx(file_name):
    '''
    separate script for getting docx content to json using docx2python pkg.
    '''
    with docx2python(file_name, html=True) as docx_content:
        docx_content_text = docx_content.text

    soup = BeautifulSoup(docx_content_text, "lxml")
    if len(soup.find_all("h1"))>=1:
        tag = ["h1", "h2", "h3","h4","h5"]
    elif len(soup.find_all("b"))>=1:
        tag = 'b'
    
    for br in soup.find_all(tag):
        br.replace_with(f"{br}")

    parsedText = soup.get_text()

    soup = BeautifulSoup(parsedText.strip(), "html.parser")
    bs = soup.find_all(tag)
    out = {}
    for each in bs:
        #print(each.text)
        out[each.text] = ''
        eachFollowingText = each.next_sibling.strip()
        #print(f'{each.text} {eachFollowingText}')
        out[each.text] = eachFollowingText.replace(each.text,'')
    return out

def get_header_paragraph(file_name):
    # get html data from file
    html_data = None
    use_doc2xpython = True

    if file_name.split('.')[-1]=='docx':
        if use_doc2xpython:
            pass
        else:
            html_data = get_html_from_docx(file_name)
    elif file_name.split('.')[-1]=='pdf':
        html_data = get_html_from_pdf(file_name)
    
    # parse html data
    html_dict = {}
    if file_name.split('.')[-1]=='docx':
        if use_doc2xpython:
            html_dict = get_json_from_docx(file_name)
        else:
            html_dict = parse_html(html_data=html_data, pattern_to_look_for=".//strong")
    elif file_name.split('.')[-1]=='pdf':
        use_file = True
        if use_file:
            #html_dict_meta= parser.from_file(file_name)
            html_dict_meta = {}
            html_dict_meta['content'] = extract_text(file_name)
        else:
            html_dict_meta = parser.from_buffer(html_data)
        #print(html_dict_meta['content'])
        for i in html_dict_meta['content'].split('\n\n'):
            if len(i)<2:
                continue
            if len(i.split('\n')) <2:
                continue
            k = i.split('\n')[0]
            v = ". ".join(i.split('\n')[1:])
            html_dict[k] = v
    
    return html_dict


if __name__=="__main__":
    #file_name = "output.docx"
    file_name = "Maternal_mortality_India_story.pdf"

    html_to_dict = get_header_paragraph(file_name)
    print(html_to_dict)
    #with open(file_name.split('.')[0]+'.json', 'w') as f:
    #    json.dump(html_to_dict,f)