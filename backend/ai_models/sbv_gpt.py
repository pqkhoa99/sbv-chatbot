import os
import sys
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger("sbv_gpt")

# Add parent directory to path to allow importing from pipeline package if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env")

# LangSmith Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "sbv-gpt"

def generate_gpt_answer(question: str):
    """
    Generate answer using GPT-5 model directly.
    """
    logger.info(f"GPT PIPELINE ({OPENAI_MODEL})")
    
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY
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
    
    answer = generate_gpt_answer(test_question)
    logger.info("Final Answer:")
    logger.info(answer)
