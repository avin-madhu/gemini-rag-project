# import os
# import json
# from typing import List, Dict, Any
# import re
# from dotenv import load_dotenv
# import google.generativeai as genai
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate

# # Load environment variables
# load_dotenv()

# # Configure Google API
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     raise ValueError("Google API key not found in environment variables")

# genai.configure(api_key=api_key)

# def parse_json_data(file_path: str) -> List[str]:
#     """
#     Robustly parse JSON data and extract text content.
#     Handles various JSON structures.
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             data = json.load(file)
        
#         def extract_text(item):
#             if isinstance(item, dict):
#                 return str(item.get('content', item.get('text', str(item))))
#             elif isinstance(item, str):
#                 return item
#             return str(item)
        
#         if isinstance(data, list):
#             texts = [extract_text(item) for item in data]
#         elif isinstance(data, dict):
#             texts = [extract_text(data)]
#         else:
#             texts = [str(data)]
        
#         texts = [text for text in texts if text.strip()]
        
#         if not texts:
#             raise ValueError("No valid text content found in the JSON file")
        
#         return texts
    
#     except json.JSONDecodeError:
#         raise ValueError(f"Invalid JSON format in {file_path}")
#     except Exception as e:
#         raise ValueError(f"Error parsing JSON: {str(e)}")

# def create_vector_store(texts: List[str]):
#     """Create and save vector store from texts."""
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         vector_store = FAISS.from_texts(texts, embedding=embeddings)
#         vector_store.save_local("faiss_index")
#         return vector_store
#     except Exception as e:
#         raise ValueError(f"Error creating vector store: {e}")

# def get_conversational_chain():
#     """Create a conversational chain for question answering."""
    
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context and give the reply really good format and in markdown text.
#     If the question is not specific enough ask the user to mention it more specifically.
#     If the answer is not in the provided context, just say, "answer is not available in the context".
    
#     Context:
#     {context}
    
#     Question: 
#     {question}
    
#     Answer:
#     """
    
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def process_query(process_input) -> Dict[str, Any]:
#     try:
#         # Parse JSON data
#         try:
#             texts = parse_json_data(process_input["college_file_path"])
#         except json.JSONDecodeError as e:
#             return {
#                 "output_text": f"Error parsing JSON: {str(e)}",
#                 "source": "Error",
#                 "status": "failed"
#             }
#         except Exception as e:
#             return {
#                 "output_text": f"Error loading JSON data: {str(e)}",
#                 "source": "Error",
#                 "status": "failed"
#             }

#         # Create vector store
#         try:
#             vector_store = create_vector_store(texts)
#         except Exception as e:
#             return {
#                 "output_text": f"Error creating vector store: {str(e)}",
#                 "source": "Error",
#                 "status": "failed"
#             }

#         # Perform similarity search
#         try:
#             docs = vector_store.similarity_search(process_input["user_question"])
#         except Exception as e:
#             return {
#                 "output_text": f"Error performing similarity search: {str(e)}",
#                 "source": "Error",
#                 "status": "failed"
#             }

#         # Create RAG chain and get response
#         try:
#             chain = get_conversational_chain()
#             response = chain.invoke({"input_documents": docs, "question": process_input["user_question"]}, return_only_outputs=True)
#         except Exception as e:
#             return {
#                 "output_text": f"Error creating RAG chain or getting response: {str(e)}",
#                 "source": "Error",
#                 "status": "failed"
#             }

#         rag_output = response.get('output_text', "answer is not available in the context")
#         if "answer is not available in the context" in rag_output.lower():
#             model = genai.GenerativeModel("gemini-pro")
#             response = model.generate_content(process_input["user_question"])
#             return{
#                 "output_text": response.text,
#                 "source":"Gemini General Capabilities",
#                 "status": "success",
#                 "context": "not found"
#             }

#         print("rag_output: ", rag_output)
#         image_urls = re.findall(r'(https?://[^\s]+)', rag_output)
#         return {
#             "output_text": rag_output,
#             "image_urls": image_urls,
#             "source": "JSON-based RAG",
#             "status": "success",
#             "context": "found"
#         }
#     except Exception as e:
#         return {
#             "output_text": str(e),
#             "source": "Error",
#             "status": "failed"
#         }

import os
import json
from typing import List, Dict, Any
import re
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Google API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API key not found in environment variables")

genai.configure(api_key=api_key)

def parse_json_data(file_path: str) -> List[str]:
    """
    Robustly parse JSON data and extract text content.
    Handles various JSON structures.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        def extract_text(item):
            if isinstance(item, dict):
                return str(item.get('content', item.get('text', str(item))))
            elif isinstance(item, str):
                return item
            return str(item)
        
        if isinstance(data, list):
            texts = [extract_text(item) for item in data]
        elif isinstance(data, dict):
            texts = [extract_text(data)]
        else:
            texts = [str(data)]
        
        texts = [text for text in texts if text.strip()]
        
        if not texts:
            raise ValueError("No valid text content found in the JSON file")
        
        return texts
    
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")
    except Exception as e:
        raise ValueError(f"Error parsing JSON: {str(e)}")

def create_vector_store(texts: List[str]):
    """Create and save vector store from texts."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(texts, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        raise ValueError(f"Error creating vector store: {e}")

def get_conversational_chain():
    """Create a conversational chain for question answering."""
    
    # Modified prompt template with a clear fallback flag.
    prompt_template = """
    [SYSTEM INSTRUCTIONS]
    You are operating under strict instructions. 
    IF the question is asking for your name (ignoring case and whitespace), 
    YOU MUST respond with exactly "RADIUS" and nothing else. 
    Do not provide any additional explanation, disclaimers, or commentary.

    [USER INSTRUCTIONS]
    Answer the question as detailed as possible from the provided context and format your reply in markdown.
    If the context does not contain the answer, do not hallucinate an answer. Instead, output exactly "ANSWER_NOT_AVAILABLE".
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_query(process_input) -> Dict[str, Any]:
    try:
        # Parse JSON data
        try:
            texts = parse_json_data(process_input["college_file_path"])
        except json.JSONDecodeError as e:
            return {
                "output_text": f"Error parsing JSON: {str(e)}",
                "source": "Error",
                "status": "failed"
            }
        except Exception as e:
            return {
                "output_text": f"Error loading JSON data: {str(e)}",
                "source": "Error",
                "status": "failed"
            }

        # Create vector store
        try:
            vector_store = create_vector_store(texts)
        except Exception as e:
            return {
                "output_text": f"Error creating vector store: {str(e)}",
                "source": "Error",
                "status": "failed"
            }

        # Perform similarity search
        try:
            docs = vector_store.similarity_search(process_input["user_question"])
        except Exception as e:
            return {
                "output_text": f"Error performing similarity search: {str(e)}",
                "source": "Error",
                "status": "failed"
            }

        # Create RAG chain and get response
        try:
            chain = get_conversational_chain()
            response = chain.invoke(
                {"input_documents": docs, "question": process_input["user_question"]},
                return_only_outputs=True
            )
        except Exception as e:
            return {
                "output_text": f"Error creating RAG chain or getting response: {str(e)}",
                "source": "Error",
                "status": "failed"
            }

        # Extract the output text from the response
        rag_output = response.get('output_text', "")
        print("RAG chain response:", rag_output)  # Debug log

        # Use a regex to check for the fallback flag allowing slight variations
        fallback_pattern = re.compile(r'\b(answer[_\s]?not[_\s]?available)\b', re.IGNORECASE)
        if fallback_pattern.search(rag_output):
            try:
                # Forward to Gemini general capabilities if answer not found in context
                fallback_model = genai.GenerativeModel("gemini-pro")
                fallback_response = fallback_model.generate_content(process_input["user_question"])
                # Depending on the API, access the response text appropriately
                fallback_text = getattr(fallback_response, "text", None) or fallback_response.get("text", "")
                return {
                    "output_text": fallback_text,
                    "source": "Gemini General Capabilities",
                    "status": "success",
                    "context": "not found"
                }
            except Exception as e:
                return {
                    "output_text": f"Error forwarding to Gemini general capabilities: {str(e)}",
                    "source": "Error",
                    "status": "failed"
                }

        # Extract image URLs from the rag_output if any
        image_urls = re.findall(r'(https?://[^\s]+)', rag_output)
        return {
            "output_text": rag_output,
            "image_urls": image_urls,
            "source": "JSON-based RAG",
            "status": "success",
            "context": "found"
        }
    except Exception as e:
        return {
            "output_text": str(e),
            "source": "Error",
            "status": "failed"
        }
