import os
import sys
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger("sbv_gemini")

# Add parent directory to path to allow importing from pipeline package if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env")

# LangSmith Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "sbv-gemini"

def generate_gemini_answer(question: str):
    """
    Generate answer using Gemini-2.5 model directly.
    """
    logger.info(f"GEMINI PIPELINE ({GEMINI_MODEL})")
    
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_template(
        """Bạn là trợ lý pháp lý hữu ích. Hãy trả lời câu hỏi sau của người dùng một cách chính xác với những căn cứ pháp lý mà bạn biết.
        
        Câu hỏi: {question}
        
        Trả lời:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        answer = chain.invoke({"question": question})
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

if __name__ == "__main__":
    test_question = "Ngân hàng có bắt buộc phải công khai thông tin về Open API trên trang thông tin điện tử không?"
    # test_question = "Khoản vay nước ngoài là gì?"
    logger.info(f"Testing with question: {test_question}")
    
    answer = generate_gemini_answer(test_question)
    logger.info("Final Answer:")
    logger.info(answer)
