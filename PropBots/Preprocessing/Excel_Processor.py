import pandas as pd

def extract_excel_data(excel_path):
    xls = pd.ExcelFile(excel_path)
    excel_data = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        excel_data[sheet_name] = df.to_dict(orient="records")
    return excel_data

