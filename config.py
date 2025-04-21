import logging
import os
import sys
from typing import Optional, List, Dict, Any, Generator
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbedding

# API配置
API_BASE_URL = "https://api.deepseek.com/v1"
API_KEY = "sk-c69bc90b7f2e4cd0b0777e676af24453"

TTS_GROUP_ID = "1913961963457614324"
TTS_API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJhbGxlbiBsYXUiLCJVc2VyTmFtZSI6ImFsbGVuIGxhdSIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTEzOTYxOTYzNDY2MDAyOTMyIiwiUGhvbmUiOiIiLCJHcm91cElEIjoiMTkxMzk2MTk2MzQ1NzYxNDMyNCIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6ImFsbGVubGF1MjYxM0BnbWFpbC5jb20iLCJDcmVhdGVUaW1lIjoiMjAyNS0wNC0yMSAwMDozODoyNyIsIlRva2VuVHlwZSI6MSwiaXNzIjoibWluaW1heCJ9.Gd8R2YFlJDC8osq0L74zjspUpSuiD9PRhaYpN4PU_mLV-JL549erxue_dES7gPBVmJZym8UCzEbUmlsRdLpOwIoAVG_Y5fnm7WT8wQYFVMuRhq5pdKH-IXdYu9-6ASL2L36mtR5dVEzVGhmfUwxVM7BbybNudpx6MJk2y3mZaLgqhsSR981lLFZf_LLqLrVqqDD1W3bnuFQUvPfULQCxgoCmM4D96kq5nBpCQLdhPK5NE_nRcM_30BMkxNbiWyPKcvjMwjfHq5U-gWtNuVrpg3aRU6I1rclYFGsJvMq3aUI00SvC0W5XBn_Mrj-OhdyfUxK5Epn99riPYVHpn4jCeQ"

# 嵌入模型配置
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# 日志配置
def setup_logging():
    """设置日志配置为控制台输出"""
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

# LLM客户端
class LLMClient:
    _instance: Optional['LLMClient'] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, api_key=None, base_url=None):
        if self._initialized:
            return
        self.api_key = api_key or API_KEY
        self.base_url = base_url or API_BASE_URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._initialized = True
        
    def chat(self, messages: List[Dict[str, Any]], temperature=0.5, stream=True) -> str:
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=temperature,
                stream=stream
            )
            return self._process_streaming_response(response, stream)
        except Exception as e:
            logger.error(f"LLM调用出错: {str(e)}")
            raise

    def chat_stream_by_sentence(self, messages: List[Dict[str, Any]], temperature=0.5, stream=True) -> Generator[str, None, str]:
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=temperature,
                stream=stream
            )
            full_response = ""
            current_sentence = ""
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    current_sentence += content
                    full_response += content
                    
                    # Split into sentences
                    sentences = sent_tokenize(current_sentence)
                    for i, sentence in enumerate(sentences[:-1]):
                        yield sentence
                    current_sentence = sentences[-1]
            
            if current_sentence.strip():
                yield current_sentence
            return full_response
        except Exception as e:
            logger.error(f"LLM调用出错: {str(e)}")
            yield f"生成回复时出错: {str(e)}"
            raise

    def _process_streaming_response(self, response, stream):
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content
        print()
        return full_response

# 嵌入模型
class EmbeddingModel:
    _instance: Optional[HuggingFaceEmbeddings] = None

    @classmethod
    def get_instance(cls) -> HuggingFaceEmbeddings:
        if cls._instance is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            logging.info(f"初始化嵌入模型: {EMBEDDING_MODEL_NAME}，使用设备: {device}")
            cls._instance = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True}
            )
        return cls._instance

# 使用示例
if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    
    llm = LLMClient()
    messages = [{"role": "user", "content": "你好"}]
    response = llm.chat(messages)
    logger.info(f"LLM响应: {response}")
    
    text = "这是一个测试文本"
    embedding_model = EmbeddingModel.get_instance()
    embedding = embedding_model.embed_query(text)
    logger.info(f"嵌入向量维度: {len(embedding)}")