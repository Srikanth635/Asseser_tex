import PyPDF2
import os
import dotenv
dotenv.load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from openai import OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    """
    Splits text into smaller chunks for processing by an LLM.

    Args:
        text (str): The full text to split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def evaluate_chunk(chunk, criteria):
    """
    Evaluates a text chunk based on user-defined criteria using LLM.

    Args:
        chunk (str): Text chunk to evaluate.
        criteria (str): Criteria for evaluation.

    Returns:
        str: Evaluation result for the chunk.
    """
    prompt = (
        f"Evaluate the following text based on the criteria: '{criteria}'.\n\n"
        f"Text:\n{chunk}\n\n"
        "Evaluation:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response["choices"][0]["message"]["content"]

def aggregate_evaluations(chunk_evaluations):
    """
    Aggregates evaluations from all chunks.

    Args:
        chunk_evaluations (list): List of evaluations for each chunk.

    Returns:
        str: Aggregated evaluation summary.
    """
    return "\n".join(chunk_evaluations)

def evaluate_pdf(pdf_path, criteria):
    """
    Evaluates a PDF document based on user-defined criteria.

    Args:
        pdf_path (str): Path to the PDF file.
        criteria (str): Criteria for evaluation.

    Returns:
        str: Aggregated evaluation summary for the document.
    """
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Split text into manageable chunks
    chunks = split_text_into_chunks(text)

    # Step 3: Evaluate each chunk
    chunk_evaluations = [evaluate_chunk(chunk, criteria) for chunk in chunks]

    # Step 4: Aggregate results
    return aggregate_evaluations(chunk_evaluations)

### For Large PDFs

def extract_text_from_large_pdf(pdf_path):
    """
    Extracts text from a large PDF file incrementally.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        generator: Yields extracted text from each page.
    """
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

def evaluate_chunks_in_batches(chunks, criteria, batch_size=5):
    """
    Evaluates text chunks in batches to handle large documents.

    Args:
        chunks (list): List of text chunks to evaluate.
        criteria (str): Criteria for evaluation.
        batch_size (int): Number of chunks to process in each batch.

    Returns:
        list: List of evaluations for each chunk.
    """
    evaluations = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_text = "\n\n".join(batch)  # Combine chunks in the batch
        prompt = (
            f"Asses the following text based on the requirement criteria: \n\n '{criteria}'."
            f"Text:\n\n{batch_text}"
            """Output Requirements: 
                1. Assign a **numerical rating (1–10)** for each criterion.
                2. Provide **detailed explanations** for each rating, including:
                   - Specific examples of strong and weak writing practices.
                   - Suggestions for improving clarity, tone, or engagement.
                   - Relevant elaborative prompts to enhance the clarity, cohesiveness in the information
                3. Summarize ratings and key findings in a **LaTeX table**.
                4. Write the report as a **LaTeX document** formatted as follows:
                   - Use section and subsection for structure.
                   - Summarize ratings and observations in a formatted table using tabular`.
                   - Include bullet points (`itemize`) to list strengths, weaknesses, and recommendations for each criterion."""
            "OUTPUT only the latex code, no additional comments or explanations are needed"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user",
                 "content": prompt}
            ],
        )

        evaluations.append(resp.choices[0].message.content)
    return evaluations

def clean_latex_code(code):
    # Split the code into lines
    lines = code.splitlines()
    # Remove unwanted markers
    cleaned_lines = [line for line in lines if not line.strip().startswith("```")]
    # Join the lines back together
    return "\n".join(cleaned_lines)

def aggregate_evaluations_with_saving(chunk_evaluations, output_path="intermediate_results.txt"):
    """
    Aggregates evaluations and saves intermediate results to a file.

    Args:
        chunk_evaluations (list): List of evaluations for each chunk.
        output_path (str): Path to save intermediate results.

    Returns:
        str: Aggregated evaluation summary.
    """
    output_file = output_path.split('.')
    for i in range(len(chunk_evaluations)):
        with open(f'{output_file[0]}_{i}.{output_file[1]}', "a") as f:
            f.write(clean_latex_code(chunk_evaluations[i]))

    with open(output_path, "w") as file:
        for eval in chunk_evaluations:
            file.write(eval + "\n\n")
    print(f"Intermediate results saved to {output_path}")
    return "\n".join(chunk_evaluations)

def evaluate_large_pdf(pdf_path, criteria, output_path="intermediate_results.txt"):
    """
    Evaluates a large PDF document based on user-defined criteria.

    Args:
        pdf_path (str): Path to the PDF file.
        criteria (str): Criteria for evaluation.
        output_path (str): Path to save intermediate results.

    Returns:
        str: Aggregated evaluation summary for the document.
    """
    # Step 1: Extract text from PDF incrementally
    text_generator = extract_text_from_large_pdf(pdf_path)

    # Step 2: Split text into manageable chunks
    chunks = split_large_text(text_generator)

    # Step 3: Evaluate chunks in batches
    chunk_evaluations = evaluate_chunks_in_batches(chunks, criteria)

    # Step 4: Aggregate results and save intermediate outputs
    return aggregate_evaluations_with_saving(chunk_evaluations, output_path)

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pdf_path = os.path.join(root, "files", "report.pdf")

    criteria_path = os.path.join(root, "src", "Asseser_GPT")
    file_name = "criteria1.txt"

    with open(os.path.join(criteria_path,file_name), 'r') as f:
        criteria = f.read()

    build_path = os.path.join(root, "src", "Asseser_GPT", "build", file_name.split(".")[0])

    try:
        os.makedirs(build_path)
        print(f"Directory '{build_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{build_path}' already exists.")

    output_path = os.path.join(build_path, "evaluation_results.tex")

    result = evaluate_large_pdf(pdf_path, criteria, output_path=output_path)
    # print(result)