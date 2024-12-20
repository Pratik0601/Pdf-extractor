import os
import json
import pdfplumber
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Use langchain's ChatOpenAI instead of langchain_openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv()  # Ensure OPENAI_API_KEY is set in .env

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

def chunk_text(text, chunk_size=1000):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Use ChatOpenAI from langchain.chat_models
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

chat_template = """
You are a document parsing assistant. Extract the following fields from the given text:
- Name
- Address
- DOB
- DL
- Officer
- Court
- Judge
- Charge
- Citation
- Issue Date
- Case
- Accident
- Cited Speed
- Posted Speed
- Offense
- Disposition
- Date

If a field is not found, return an empty string for that field.
Respond in JSON format with keys exactly as listed.

TEXT:
{chunk}
"""

prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(chat_template)
])

chain = prompt | llm

def parse_llm_result(result):
    # result is an AIMessage object, use result.content
    content = result.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "Name": "",
            "Address": "",
            "DOB": "",
            "DL": "",
            "Officer": "",
            "Court": "",
            "Judge": "",
            "Charge": "",
            "Citation": "",
            "Issue Date": "",
            "Case": "",
            "Accident": "",
            "Cited Speed": "",
            "Posted Speed": "",
            "Offense": "",
            "Disposition": "",
            "Date": ""
        }
    return data

def validate_date(value):
    if not value:
        return value
    for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y"]:
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value

def validate_fields(df):
    if "Issue Date" in df.columns:
        df["Issue Date"] = df["Issue Date"].apply(validate_date)
    return df

def export_results(df, csv_name="extracted_data.csv", excel_name="extracted_data.xlsx"):
    df.to_csv(csv_name, index=False)
    df.to_excel(excel_name, index=False)
    print(f"Data exported to {csv_name} and {excel_name}")

if __name__ == "__main__":
    file_path = r"E:\2024.pdf"
    raw_text = extract_text_from_pdf(file_path)
    chunks = chunk_text(raw_text, chunk_size=1000)

    results = []
    for c in chunks:
        result = chain.invoke({"chunk": c})
        data = parse_llm_result(result)
        results.append(data)

    df = pd.DataFrame(results)
    df = validate_fields(df)
    export_results(df, "extracted_data.csv", "extracted_data.xlsx")
