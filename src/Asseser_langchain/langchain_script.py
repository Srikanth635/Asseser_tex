from langchain.chains.sequential import SequentialChain
from langchain_openai.llms import OpenAI
import os
from prompt.prompt_template import *
import tiktoken
import PyPDF2
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter

root_dir = os.getcwd()
docs_dir = os.path.join(root_dir, 'pdf_files')
prompts_dir = os.path.join(root_dir, 'prompt')

def document_reader(doc_path):
    with open(doc_path, 'r') as f:
        prompt = f.read()
    return prompt

def token_counter(model, messages):
    final_prompt = "\n\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
    )
    encoding = tiktoken.encoding_for_model(model)
    token_count = 0
    for message in messages:
        # Add the role and the content
        token_count += len(encoding.encode(message['role']))
        token_count += len(encoding.encode(message['content']))
        # Add extra tokens for message formatting, such as separators
        token_count += 4  # Approximation for formatting tokens
    # Print the total token count
    print(f"Total tokens in the prompt: {token_count}")
    return token_count

def extract_text_from_pdf_page_by_page(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            yield page.extract_text()

def split_large_text(text_generator, chunk_size=1000, chunk_overlap=100):
    """
    Splits large text into smaller chunks dynamically.

    Args:
        text_generator (generator): Generator yielding text from the PDF.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    for text in text_generator:
        chunks.extend(splitter.split_text(text))
    return chunks

if __name__ == '__main__':

    content_file = document_reader(os.path.join(docs_dir, 'report.pdf'))

    prompt = document_reader(os.path.join(prompts_dir, 'prompt1.txt'))
    print(prompt)