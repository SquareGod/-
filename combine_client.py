from typing import Iterable
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import AddableDict
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from knowledge import MyKnowledge
from models import ALI_TONGYI_MAX_MODEL, get_lc_model_client

# 知识库问答系统提示词
qa_system_prompt = """你是一名知识问答助手，
              你将使用检索到的上下文来回答问题。如果你不知道答案，就说你没有找到答案。 "
              "\n\n"
              "{context}" 
        """

# 构建知识库问答Prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# 构建普通问答Prompt
normal_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个帮助人们解答各种问题的助手。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# 流式输出解析器：将模型返回的chunk转换为指定格式
def streaming_parse(chunks: Iterable[AIMessageChunk]):
    for chunk in chunks:
        yield AddableDict({'answer': chunk.content})

# 组合客户端：整合大模型交互、聊天历史、知识库检索能力
class CombineClient(MyKnowledge):
    # 聊天历史存储
    __chat_history = ChatMessageHistory()

    # 获取处理链（根据是否指定知识库，返回RAG链/普通问答链）
    def get_chain(self, collection, model, max_length, temperature):
        retriever = None
        # 若指定知识库，初始化检索器
        if collection:
            retriever = self.get_retrievers(collection)

        # 聊天历史最多保留3轮（6条消息：3问3答）
        if len(self.__chat_history.messages) >= 6:
            self.__chat_history.messages = self.__chat_history.messages[-6:]

        # 获取大模型客户端
        chat = get_lc_model_client(model=model, max_tokens=max_length, temperature=temperature)

        # 构建RAG链（带知识库检索）
        if retriever:
            question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        # 构建普通问答链（无知识库）
        else:
            rag_chain = normal_prompt | chat | streaming_parse

        # 为链添加聊天历史管理能力
        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: self.__chat_history,  # 绑定聊天历史存储
            input_messages_key="input",              # 输入问题的key
            history_messages_key="chat_history",     # 聊天历史的key
            output_messages_key="answer",            # 输出答案的key
        )
        return chain_with_history

    # 同步调用：获取完整回答
    def invoke(self, question, collection, model=ALI_TONGYI_MAX_MODEL, max_length=256, temperature=1):
        return self.get_chain(collection, model, max_length, temperature).invoke(
            {"input": question},
            {"configurable": {"session_id": "unused"}},
        )

    # 流式调用：逐段返回回答
    def stream(self, question, collection, model=ALI_TONGYI_MAX_MODEL, max_length=256, temperature=1):
        return self.get_chain(collection, model, max_length, temperature).stream(
            {"input": question},
            {"configurable": {"session_id": "unused"}},
        )

    # 清空聊天历史
    def clear_history(self) -> None:
        self.__chat_history.clear()

    # 获取聊天历史消息
    def get_history_message(self):
        return self.__chat_history.messages
