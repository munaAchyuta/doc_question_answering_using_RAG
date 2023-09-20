import pdfplumber
from dataclasses import dataclass


@dataclass
class DocPdfplumber:
    # Import the PDF.
    path: str = ""
    pdf: str = None

    def __post_init__(self):
        self.pdf = pdfplumber.open(self.path)

    def curves_to_edges(self,cs):
        """See https://github.com/jsvine/pdfplumber/issues/127"""
        edges = []
        for c in cs:
            edges += pdfplumber.utils.rect_to_edges(c)
        return edges

    # Table settings.
    def table_settings(self,p,strategy='explicit'):
        self.ts = {
                    "vertical_strategy": strategy,
                    "horizontal_strategy": strategy,
                    "explicit_vertical_lines": self.curves_to_edges(p.curves + p.edges),
                    "explicit_horizontal_lines": self.curves_to_edges(p.curves + p.edges),
                    "intersection_y_tolerance": 10,
                }
    
    def get_table(self,p):
        # Get the bounding boxes of the tables on the page.
        tables = [table.extract() for table in p.find_tables(table_settings=self.ts)]

        return tables
    
    def get_bboxes(self,p):
        self.bboxes = [table.bbox for table in p.find_tables(table_settings=self.ts)]

    def not_within_bboxes(self,obj):
        """Check if the object is in any of the table's bbox."""
        def obj_in_bbox(_bbox):
            """See https://github.com/jsvine/pdfplumber/blob/stable/pdfplumber/table.py#L404"""
            v_mid = (obj["top"] + obj["bottom"]) / 2
            h_mid = (obj["x0"] + obj["x1"]) / 2
            x0, top, x1, bottom = _bbox
            return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
        return not any(obj_in_bbox(__bbox) for __bbox in self.bboxes)

    def get_pdf_page_info(self,):
        data_pages = []
        # Load pages.
        for each_page_index in range(len(self.pdf.pages)):
            p = self.pdf.pages[each_page_index]
            try:
                self.table_settings(p,strategy='lines')
                self.get_bboxes(p)
            except:
                print('failed with strategy=lines')
                try:
                    self.table_settings(p,strategy='explicit')
                    self.get_bboxes(p)
                except:
                    print('failed with strategy=explicit')
                    self.table_settings(p,strategy='text')
                    self.get_bboxes(p)
            
            page_number = each_page_index + 1
            tables = self.get_table(p)

            tables_text = ''
            for each_table in tables:
                tables_text += ' '.join([j for i in each_table for j in i if j is not None])

            page_without_table = p.filter(self.not_within_bboxes).extract_text()
            data_pages.append({"page_number":page_number,
                               "table_present": True if len(tables) != 0 else False,
                               "tables":tables,
                               "tables_text":tables_text,
                               "page_without_table":page_without_table})
            
        return data_pages


if __name__ == "__main__":
    path = r"C:\Users\achyuta.sahoo\Documents\work\NL_to_SQL\prod\document_retriever_bkp\document_retriever\data\pdf_docs\Evaluaserv SOW 12 Data Science Pilot Team.pdf"
    doc_pdfplumber = DocPdfplumber(path)
    data_pages = doc_pdfplumber.get_pdf_page_info()
    for each_page in data_pages:
        print(each_page)
        print("\n=====\n")