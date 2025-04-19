import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
import re

# Assuming document_processor.py is in the same directory
from document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
except Exception as e:
    st.error(f"Failed to configure Google AI: {e}")
    st.stop()


VECTOR_STORE_DIR = "vector_stores"
if not os.path.exists(VECTOR_STORE_DIR):
    try:
        os.makedirs(VECTOR_STORE_DIR)
    except OSError as e:
        st.error(f"Failed to create vector store directory '{VECTOR_STORE_DIR}': {e}")
        st.stop()


@st.cache_resource
def load_vector_store(store_path):
    if not os.path.exists(store_path):
         logging.error(f"Vector store path not found: {store_path}")
         st.error(f"Vector store not found. Please ensure the PDF was processed correctly.")
         return None
    try:
         logging.info(f"Loading vector store from: {store_path}")
         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
         vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
         logging.info(f"Vector store loaded successfully from {store_path}")
         return vector_store
    except Exception as e:
         logging.error(f"Error loading vector store from {store_path}: {e}")
         st.error(f"Failed to load the document's data. Please try reprocessing the PDF. Error: {e}")
         return None

@st.cache_resource
def get_qa_chain():
     prompt_template = """
    You are an AI assistant expert at analyzing text extracted from documents. Your task is to answer questions based ONLY on the provided text Context.
    Be aware that the Context comes from PDF text extraction and may not preserve original formatting, especially for tables or figures.

    Instructions:
    1. Carefully review the Context provided below.
    2. Answer the user's Question using ONLY information explicitly found within the Context.
    3. If the Question asks about a specific element like a table or figure (e.g., "what is table 7", "provide figure 3", "show data from table X or figure "):
        a. First, describe the element's purpose, content, or main findings as stated in the Context.
        b. Then, if the Context includes specific data points, list entries, or row/column information clearly associated with that element, extract and present that information clearly.
        c. If the Context only describes the element but doesn't contain its specific data content in a usable format, clearly state that only a description is available in the text. Do NOT attempt to recreate tables or figures from poorly formatted text.
    4. If the information needed to answer the Question is not present in the Context at all, state clearly:
    "Based on the provided text, the answer to your question is not available in this document."
    5. Do NOT add external knowledge or information not present in the Context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
     try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.15)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        logging.info("QA Chain Initialized.")
        return chain
     except Exception as e:
         logging.error(f"Failed to initialize QA chain: {e}")
         st.error(f"Could not initialize the AI model for Q&A. Error: {e}")
         return None


def sanitize_filename(filename):
    s = re.sub(r'[^\w\s-]', '', filename).strip()
    s = re.sub(r'[-\s]+', '-', s)
    return s


st.set_page_config(layout="wide")
st.title("📄 Chat with your PDF Document")
st.markdown("Upload a PDF, wait for processing, and then ask questions about its content.")


if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "vector_store_path" not in st.session_state:
    st.session_state.vector_store_path = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


uploaded_file = st.file_uploader("1. Upload your PDF File", type="pdf", key="pdf_uploader")

if uploaded_file:
    temp_pdf_path = None
    current_file_name = uploaded_file.name
    new_file_uploaded = (st.session_state.uploaded_file_name != current_file_name)

    if new_file_uploaded:
        st.info(f"New file detected: '{current_file_name}'. Processing...")
        st.session_state.pdf_processed = False
        st.session_state.vector_store_path = None
        st.session_state.uploaded_file_name = current_file_name


    if not st.session_state.pdf_processed:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_pdf_path = tmp_file.name

            safe_pdf_name = sanitize_filename(os.path.splitext(current_file_name)[0])
            vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{safe_pdf_name}_faiss_index")

            processor = DocumentProcessor()

            with st.spinner(f"Processing '{current_file_name}'... This might take a moment."):
                logging.info(f"Starting processing for {current_file_name}")
                pages_content = processor.extract_text_with_pages(temp_pdf_path)
                documents = processor.split_text_into_chunks(pages_content)
                processor.create_vector_store(documents, vector_store_path)

            st.session_state.pdf_processed = True
            st.session_state.vector_store_path = vector_store_path
            st.success(f"✅ Successfully processed '{current_file_name}'!")
            logging.info(f"Finished processing {current_file_name}")

        except (ValueError, RuntimeError, FileNotFoundError, OSError) as e:
            st.error(f"Error processing PDF: {e}")
            logging.error(f"Processing failed for {current_file_name}: {e}")
            st.session_state.pdf_processed = False
            st.session_state.vector_store_path = None
            st.session_state.uploaded_file_name = None # Reset to allow re-upload/retry
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logging.error(f"Unexpected error during processing of {current_file_name}: {e}")
            st.session_state.pdf_processed = False
            st.session_state.vector_store_path = None
            st.session_state.uploaded_file_name = None
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                    logging.info(f"Removed temporary file: {temp_pdf_path}")
                except OSError as e:
                    logging.warning(f"Could not remove temporary file {temp_pdf_path}: {e}")


if st.session_state.pdf_processed and st.session_state.vector_store_path:
    st.markdown("---")
    st.header("2. Ask Questions About the Document")

    vector_store = load_vector_store(st.session_state.vector_store_path)
    qa_chain = get_qa_chain()

    if vector_store and qa_chain:
        user_question = st.text_input("Enter your question here:", key="user_question")

        if user_question:
            with st.spinner("Searching for the answer..."):
                try:
                    logging.info(f"Performing similarity search for question: {user_question}")
                    relevant_docs = vector_store.similarity_search(user_question, k=10)

                    if not relevant_docs:
                        st.warning("Could not find relevant segments in the document for your question.")
                        logging.warning(f"No relevant documents found for question: {user_question}")
                    else:
                        logging.info(f"Invoking QA chain with {len(relevant_docs)} documents.")
                        response = qa_chain.invoke(
                               {"input_documents": relevant_docs, "question": user_question}
                           )
                        answer = response.get("output_text", "Error: Could not retrieve answer from response object.")
                        logging.info("Received response from QA chain.")

                        st.markdown("### Answer")
                        st.markdown(answer)

                        with st.expander("Show Context Used"):
                             st.markdown("The AI generated the answer based on the following text segments:")
                             for i, doc in enumerate(relevant_docs):
                                 page_num = doc.metadata.get('page', 'N/A')
                                 st.markdown(f"**Segment {i+1} (from Page {page_num}):**")
                                 st.caption(doc.page_content)


                except Exception as e:
                     st.error(f"An error occurred while answering the question: {e}")
                     logging.error(f"Error during QA invocation or processing: {e}")
    else:
        st.error("Required components (vector store or QA chain) failed to load. Please try reprocessing the PDF.")

elif uploaded_file and not st.session_state.pdf_processed:
     st.warning("⏳ PDF processing is pending or encountered an error. Please wait or check error messages above.")

else:
     st.info("👋 Welcome! Please upload a PDF file using the uploader above to get started.")