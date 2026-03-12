import hashlib
import os
import shutil
from typing import Optional

from langchain.indexes import SQLRecordManager
from langchain.retrievers import ContextualCompressionRetriever, RePhraseQueryRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainFilter, CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.document_loaders import BaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index

from custom_loader import MyCustomLoader
from models import get_lc_model_client, get_ali_embeddings, get_ali_rerank
from loguru import logger

# 设置知识库 向量模型 重排序模型的路径
KNOWLEDGE_DIR = './chroma/knowledge/'
embedding_model = get_ali_embeddings()

class MyKnowledge:
    """
    知识库管理模块
    """
    # 向量化模型
    __embeddings = embedding_model
    logger.info(f"当前嵌入模型: {__embeddings.model}")

    __retrievers = {}
    __llm = get_lc_model_client()

    def upload_knowledge(self, temp_file):
        """
        处理原始文档的上传，并负责启动文档索引过程
        """
        file_name = os.path.basename(temp_file)
        file_path = os.path.join(KNOWLEDGE_DIR, file_name)
        # 如果文件不存在就copy
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            shutil.copy(temp_file, file_path)

        import gradio as gr
        return None, gr.update(choices=self.load_knowledge())

    def load_knowledge(self):
        # exist_ok=True目标目录已存在的情况下不会抛出异常。
        # 这意味着如果目录已经存在，os.makedirs不会做任何事情，也不会报错
        os.makedirs(os.path.dirname(KNOWLEDGE_DIR), exist_ok=True)

        # 知识库默认为空
        collections = [None]
        logger.info(f"当前知识库文件列表: {os.listdir(KNOWLEDGE_DIR)}")

        for file in os.listdir(KNOWLEDGE_DIR):
            # 将知识库进行添加
            collections.append(file)

            # 得到知识库的路径
            file_path = os.path.join(KNOWLEDGE_DIR, file)
            logger.info(f"文件路径: {file_path}")

            # 简单化处理知识库名称，由文件名md5编码得到
            collection_name = get_md5(file)
            logger.info(f"知识库名称: {collection_name}")
            logger.info(f"self.__retrievers: {self.__retrievers}")

            if collection_name in self.__retrievers:
                continue

            # 创建对应加载器
            loader = MyCustomLoader(file_path)

            # 检索
            self.__retrievers[collection_name] = create_indexes(collection_name, loader, self.__embeddings)
            logger.info(f"检索器详情: {self.__retrievers}")

        return collections

    def get_retrievers(self, collection):
        collection_name = get_md5(collection)
        logger.info(f"知识库名字md5: {collection_name}")
        if collection_name not in self.__retrievers:
            logger.info(f"self.__retrievers:: {self.__retrievers:}")
            return None

        retriever = self.__retrievers[collection_name]
        logger.info(f"get_retrievers中: {retriever}")
        """
        ContextualCompressionRetriever:在上下文中压缩和优化检索结果
        结合了基础压缩器（base_compressor）和基础检索器（base_retriever），以减少不相关信息，返回更为精炼的检索结果。
        """
        compression_retriever = ContextualCompressionRetriever(
            # 初始化一个 LLMChainFilter实例。该实例会使用大语言模型来执行复杂的文本过滤逻辑。
            base_compressor=LLMChainFilter.from_llm(self.__llm),
            # 利用语言模型对查询进行重述或重新表述，以提取问题的关键元素，从而优化检索过程。
            base_retriever=RePhraseQueryRetriever.from_llm(retriever, self.__llm)
        )

        # 用于对检索结果进行重新排序
        rerank_retriever = get_ali_rerank(top_n=3)

        # 结合了基础检索器和压缩器的功能，先从数据库中检索候选文档，然后对这些文档进行压缩或过滤，以返回最相关的结果。
        final_retriever = ContextualCompressionRetriever(
            base_compressor=rerank_retriever, base_retriever=compression_retriever
        )
        logger.debug(f"最终检索器为: {final_retriever}")
        return final_retriever

# 创建索引
def create_indexes(collection_name: str, loader: BaseLoader, embedding_function: Optional[Embeddings] = None):
    # 初始化Chroma数据库
    db = Chroma(collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=os.path.join('./chroma', collection_name))

    # 初始化记录管理器:管理与文档相关的元数据和检索信息，并将这些数据存储在一个SQL数据库中
    record_manager = SQLRecordManager(
        f"chromadb/{collection_name}", db_url="sqlite:///db/record_manager_cache.db"
    )
    """在文档被索引到Chroma数据库之前，SQLRecordManager会管理这些文档的索引记录。它确保文档的索引状态被正确记录，避免重复索引或遗漏。"""
    logger.info(f"record_manager: {record_manager}")
    # 初始化数据库表结构
    record_manager.create_schema()
    """
    在你开始加载和索引文档之前，调用create_schema() 方法，确保用于存储文档元数据和索引信息的数据库表已经存在。
    如果是第一次运行这个脚本，create_schema()方法会自动创建这些表。对于后续运行，如果表已经存在，则该方法不会重复创建，而是直接通过。
    """
    logger.info("准备进行文件的加载切分....")
    documents = loader.load()
    logger.info(f"文档切分数量: {len(documents)}....")
    logger.debug(f"documents: {documents}")

    # 将加载的文档索引到数据库中
    r = index(documents, record_manager, db, cleanup="full", source_id_key="source")
    logger.info(f"文档索引结果为: {r}")
    """
    num_added: 0 表示没有新文档被添加到数据库中。
    num_updated: 0 表示没有文档被更新。
    num_skipped: 8 表示有8个文档被跳过，没有被索引到数据库中。 提高索引效率
    num_deleted: 0 表示没有文档被删除。
    """
    '''混合检索，将稀疏检索器（如BM25）与密集检索器（如嵌入相似性）相结合。
    稀疏检索器擅长根据关键字查找相关文档，而密集检索器擅长根据语义相似性查找相关文档。'''
    ensemble_retriever = EnsembleRetriever(
        # 返回最相似的3个文档
        retrievers=[db.as_retriever(search_kwargs={"k": 3}), BM25Retriever.from_documents(documents)]
    )
    # logger.info(f"ensemble_retriever: {ensemble_retriever}")
    return ensemble_retriever

def get_md5(input_string):
    # 创建一个 md5 哈希对象
    hash_md5 = hashlib.md5()
    # 需要确保输入字符串是字节串，因此如果它是字符串，则需要编码为字节串
    hash_md5.update(input_string.encode('utf-8'))
    # 获取十六进制的哈希值
    return hash_md5.hexdigest()
