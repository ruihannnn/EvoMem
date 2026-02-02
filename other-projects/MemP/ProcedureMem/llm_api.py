import time
from openai import OpenAI
import json
from retry import retry
import os
from pathlib import Path
from dotenv import load_dotenv
from utu.utils.path import DIR_ROOT

env_path = DIR_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


API_KEY = os.getenv("UTU_LLM_API_KEY")
API_BASE = os.getenv("UTU_LLM_BASE_URL")
MODEL_NAME = os.getenv("UTU_LLM_MODEL")

# Embedding 配置 - 使用 .env 中的 EMBEDDING_LLM_* 环境变量
EMBEDDING_LLM_TYPE = os.getenv("EMBEDDING_LLM_TYPE", "openai")
EMBEDDING_LLM_MODEL = os.getenv("EMBEDDING_LLM_MODEL", "text-embedding-3-small")
EMBEDDING_LLM_BASEURL = os.getenv("EMBEDDING_LLM_BASEURL")
EMBEDDING_LLM_API_KEY = os.getenv("EMBEDDING_LLM_API_KEY")
TEMPERATURE = 0.2
TOP_P = 1
MAX_TOKENS = 4096

# Lazy initialization of client to avoid requiring API_KEY at import time
client = None

def get_client():
    global client
    if client is None:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    return client

def get_response(messages):
    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )
    if not hasattr(response, "error"):
        return response.choices[0].message.content
    return response.error.message

@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def get_llm_response(messages, is_string=False):
    ans = get_response(messages)
    if is_string:
        return ans
    else:
        cleaned_text = ans.strip("`json\n").strip("`\n").strip("```\n")
        ans = json.loads(cleaned_text)
        return ans

from langchain_openai import OpenAIEmbeddings

def get_embedding_model():
    """
    获取 embedding 模型，从 .env 文件读取配置：
    - EMBEDDING_LLM_TYPE: 模型类型（默认 openai）
    - EMBEDDING_LLM_MODEL: 模型名称（默认 text-embedding-3-small）
    - EMBEDDING_LLM_BASEURL: API base URL
    - EMBEDDING_LLM_API_KEY: API key
    """
    embedding = OpenAIEmbeddings(
        model=EMBEDDING_LLM_MODEL,
        openai_api_key=EMBEDDING_LLM_API_KEY,
        openai_api_base=EMBEDDING_LLM_BASEURL,
        max_retries=10
    )
    return embedding

if __name__ == "__main__":
    # 加载 .env 文件中的环境变量
    message = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    print(get_response(message))
    # test embedding
    embedding = get_embedding_model()
    print(embedding.embed_query("Hello, how are you?"))