import os

from langchain_community.document_compressors import DashScopeRerank
from langchain_openai import ChatOpenAI
import inspect
from langchain_community.embeddings import DashScopeEmbeddings

ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max-latest"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"  #云帆老师的免费额度最近到期时间：2025-07-26
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest"  #云帆老师的免费额度最近到期时间：2025-09-22
ALI_TONGYI_EMBEDDING_MODEL = "text-embedding-v3"
ALI_TONGYI_RERANK_MODEL = "gte-rerank-v2"  #云帆老师的免费额度最近到期时间：2025-09-14

def get_lc_model_client(api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME), base_url=ALI_TONGYI_URL
                        , model=ALI_TONGYI_MAX_MODEL, temperature=0.7, max_tokens=8000,verbose=False, debug=False):
    """
        通过LangChain获得指定平台和模型的客户端，设定的默认平台和模型为阿里百炼qwen-max-latest
        也可以通过传入api_key，base_url，model三个参数来覆盖默认值
        verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    """
    function_name = inspect.currentframe().f_code.co_name
    if (verbose):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature}")
    if (debug):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature},key：{api_key}")
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=temperature,max_tokens=max_tokens)

def get_ali_embeddings():
    """通过LangChain获得一个阿里通义千问嵌入模型的实例"""
    return DashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING_MODEL, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
    )

def get_ali_rerank(top_n=10):
    '''
    通过LangChain获得一个阿里重排序模型的实例
    :return: 阿里通义千问嵌入模型的实例
    '''
    return DashScopeRerank(
        model=ALI_TONGYI_RERANK_MODEL, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
        top_n=top_n
)


