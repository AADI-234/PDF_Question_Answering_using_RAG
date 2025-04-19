import os
from PyPDF2 import PdfReader, errors
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    def __init__(self):
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def extract_text_with_pages(self, pdf_path):
        pages_content = []
        try:
            pdf_reader = PdfReader(pdf_path)
            logging.info(f"Opened PDF: {pdf_path}")
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        processed_text = " ".join(page_text.splitlines())
                        pages_content.append({"page": i + 1, "content": processed_text})
                except Exception as page_ex:
                    logging.warning(f"Could not extract text from page {i+1} in {pdf_path}: {page_ex}")
                    continue # Skip corrupted pages if possible
        except errors.PdfReadError as e:
            logging.error(f"Failed to read PDF file {pdf_path}: {e}")
            raise ValueError(f"Invalid or corrupted PDF file: {os.path.basename(pdf_path)}") from e
        except FileNotFoundError as e:
            logging.error(f"PDF file not found at path: {pdf_path}")
            raise e
        except Exception as e:
             logging.error(f"An unexpected error occurred during PDF processing {pdf_path}: {e}")
             raise ValueError(f"Could not process PDF file: {os.path.basename(pdf_path)}") from e

        if not pages_content:
             logging.warning(f"No text could be extracted from PDF: {pdf_path}")
             raise ValueError(f"No text content found in the PDF: {os.path.basename(pdf_path)}")

        logging.info(f"Extracted text from {len(pages_content)} pages.")
        return pages_content

    def split_text_into_chunks(self, pages_content):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        all_docs = []
        for page_info in pages_content:
            page_chunks = text_splitter.split_text(page_info["content"])
            for chunk in page_chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={"page": page_info["page"]}
                )
                all_docs.append(doc)

        logging.info(f"Split text into {len(all_docs)} chunks.")
        return all_docs

    def create_vector_store(self, documents, save_path):
        if not documents:
            logging.error("Cannot create vector store: No documents provided.")
            raise ValueError("Cannot create vector store from empty document list.")
        try:
            logging.info(f"Creating vector store at: {save_path}")
            vector_store = FAISS.from_documents(documents, embedding=self.embeddings_model)
            vector_store.save_local(save_path)
            logging.info(f"Vector store saved successfully to {save_path}")
        except Exception as e:
            logging.error(f"Failed to create or save vector store at {save_path}: {e}")
            raise RuntimeError(f"Could not create vector store: {e}") from e