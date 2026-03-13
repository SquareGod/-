import hashlib
import os
import shutil
from typing import Optional
import gradio as gr
from langchain_classic.indexes import SQLRecordManager
from langchain_classic.retrievers import ContextualCompressionRetriever, RePhraseQueryRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import LLMChainFilter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.document_loaders import BaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index
from custom_loader import MyCustomLoader
from models import get_lc_model_client, get_ali_embeddings, get_ali_rerank

# 设置知识库、向量模型、重排序模型的路径
KNOWLEDGE_DIR = './chroma/knowledge/'
embedding_model = get_ali_embeddings()

class MyKnowledge:
    """
    知识库管理模块：负责文档上传、索引构建、检索器生成
    """
    # 向量化模型实例
    __embeddings = embedding_model
    # 检索器缓存（key：知识库MD5标识，value：检索器实例）
    __retrievers = {}
    # 大模型客户端（用于检索优化）
    __llm = get_lc_model_client()

    def upload_knowledge(self, temp_file):
        """
        处理原始文档的上传，并启动文档索引过程
        """
        file_name = os.path.basename(temp_file)
        file_path = os.path.join(KNOWLEDGE_DIR, file_name)
        # 文件不存在时复制到知识库目录
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            shutil.copy(temp_file, file_path)
        return None, gr.update(choices=self.load_knowledge())

    def load_knowledge(self):
        """
        加载所有知识库文件，为未处理的文件创建检索器
        """
        os.makedirs(os.path.dirname(KNOWLEDGE_DIR), exist_ok=True)
        collections = []

        for file in os.listdir(KNOWLEDGE_DIR):
            collections.append(file)
            file_path = os.path.join(KNOWLEDGE_DIR, file)
            # 生成知识库唯一标识（文件名MD5）
            collection_name = get_md5(file)

            # 已创建检索器则跳过
            if collection_name in self.__retrievers:
                continue

            # 创建文档加载器并构建索引
            loader = MyCustomLoader(file_path)
            self.__retrievers[collection_name] = create_indexes(collection_name, loader, self.__embeddings)

        # 无知识库时返回提示
        if not collections:
            collections = ["无可用知识库"]
        return collections

    def get_retrievers(self, collection):
        """
        获取优化后的检索器：结合查询重述、内容压缩、结果重排序
        """
        collection_name = get_md5(collection)
        if collection_name not in self.__retrievers:
            return None

        retriever = self.__retrievers[collection_name]

        # 上下文压缩检索器：LLM过滤无关内容 + 查询重述优化检索
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(self.__llm),
            base_retriever=RePhraseQueryRetriever.from_llm(retriever, self.__llm)
        )

        # 检索结果重排序（取Top3）
        rerank_retriever = get_ali_rerank(top_n=3)

        # 最终检索器：压缩过滤 + 重排序，返回最相关结果
        final_retriever = ContextualCompressionRetriever(
            base_compressor=rerank_retriever, base_retriever=compression_retriever
        )
        return final_retriever


def create_indexes(collection_name: str, loader: BaseLoader, embedding_function: Optional[Embeddings] = None):
    """
    构建知识库索引：初始化向量库 + 加载文档 + 创建混合检索器
    """
    # 初始化Chroma向量数据库
    db = Chroma(collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=os.path.join('./chroma', collection_name))

    # 初始化记录管理器：管理文档索引状态，避免重复/遗漏
    record_manager = SQLRecordManager(
        f"chromadb/{collection_name}", db_url="sqlite:///db/record_manager_cache.db"
    )
    # 创建记录管理器表结构（首次运行自动创建，后续无操作）
    record_manager.create_schema()

    # 加载并切分文档
    documents = loader.load()

    # 文档索引（全量清理旧数据，保证索引最新）
    index(documents, record_manager, db, cleanup="full", source_id_key="source")

    # 混合检索器：向量检索（语义相似） + BM25检索（关键词匹配），各返回Top3
    ensemble_retriever = EnsembleRetriever(
        retrievers=[db.as_retriever(search_kwargs={"k": 3}), BM25Retriever.from_documents(documents)]
    )
    return ensemble_retriever


def get_md5(input_string):
    """
    生成字符串的MD5哈希值，用于知识库唯一标识
    """
    hash_md5 = hashlib.md5()
    hash_md5.update(input_string.encode('utf-8'))
    return hash_md5.hexdigest()