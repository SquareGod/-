from langchain_community.document_loaders import CSVLoader, TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader, \
    UnstructuredMarkdownLoader
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.file_utils.filetype import FileType, detect_filetype
from loguru import logger

class MyCustomLoader(BaseLoader):
    """
    文档加载和分割模块
    """
    # 支持加载的文件类型
    file_type = {
        FileType.CSV: (CSVLoader, {'autodetect_encoding': True}),
        FileType.TXT: (TextLoader, {'autodetect_encoding': True}),
        FileType.DOC: (UnstructuredWordDocumentLoader, {}),
        FileType.DOCX: (UnstructuredWordDocumentLoader, {}),
        FileType.PDF: (PyPDFLoader, {}),
        FileType.MD: (UnstructuredMarkdownLoader, {})
    }

    # 初始化方法，设置文档加载器和文本分割器
    def __init__(self, file_path: str):
        loader_class, params = self.file_type[detect_filetype(file_path)]
        logger.info(f"本文档[{file_path}]需使用文档加载器: {loader_class}")
        self.loader: BaseLoader = loader_class(file_path, **params)
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=500,
            chunk_overlap=200,
            length_function=len,
        )

    def lazy_load(self):
        # 文档的切分加载
        return self.loader.load_and_split(self.text_splitter)

    def load(self):
        # 加载
        return self.lazy_load()
