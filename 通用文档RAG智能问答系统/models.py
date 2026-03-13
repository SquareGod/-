import os
import inspect

from langchain_community.document_compressors import DashScopeRerank
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

# 阿里百炼API相关配置
ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 模型名称常量
ALI_TONGYI_MAX_MODEL = "qwen-max-latest"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest"
ALI_TONGYI_EMBEDDING_MODEL = "text-embedding-v3"
ALI_TONGYI_RERANK_MODEL = "gte-rerank-v2"

def get_lc_model_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME), base_url=ALI_TONGYI_URL
                        , model=ALI_TONGYI_MAX_MODEL, temperature=0.7, max_tokens=8000,verbose=False, debug=False):
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=temperature,max_tokens=max_tokens)

def get_ali_embeddings():
    """
    获取阿里通义千问文本嵌入模型实例
    :return: DashScopeEmbeddings实例
    """
    return DashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING_MODEL, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
    )

def get_ali_rerank(top_n=10):
    """
    获取阿里重排序模型实例
    :param top_n: 返回最相关的top_n条结果（默认10）
    :return: DashScopeRerank实例
    """
    return DashScopeRerank(
        model=ALI_TONGYI_RERANK_MODEL, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
        top_n=top_n
    )