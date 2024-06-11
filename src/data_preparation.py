import pdfplumber
import os
import json

def extract_text_from_pdfs(folder_path):
    all_texts = []
    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith('.pdf'):
            with pdfplumber.open(os.path.join(folder_path, pdf_file)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_texts.append(text)
    return all_texts

def preprocess_text(text):
    text = text.lower()
    punctuation_to_remove = '.,;:"\''
    text = text.translate(str.maketrans('', '', punctuation_to_remove))
    text = text.translate(str.maketrans('', '', '0123456789'))
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = ' '.join(text.split())
    return text

def save_processed_texts(folder_path, output_file):
    texts = extract_text_from_pdfs(folder_path)
    processed_texts = [preprocess_text(text) for text in texts]

    with open(output_file, 'w') as f:
        json.dump(processed_texts, f)

    print("Processed and saved texts successfully.")

if __name__ == "__main__":
    folder_path = 'path/to/your/pdf/folder'  # Update this to the correct relative or absolute path
    output_file = 'data/processed_aristotle_texts.json'
    save_processed_texts(folder_path, output_file)
