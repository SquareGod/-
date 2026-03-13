from datasets import Dataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate

from combine_client import CombineClient
from models import get_lc_model_client, get_ali_embeddings

# 初始化知识库+大模型交互客户端并加载知识库
llm = CombineClient()
llm.load_knowledge()

# 调用大模型获取指定知识库的问答结果
response = llm.invoke("电子鼻发展过程", "电子鼻的算法设计综述.pdf")
print("大模型答复：", response['answer'])

# 请假流程真实标准答案（用于评估）
truth = '''1.电子鼻的算法设计以信号预处理、特征提取 / 降维、模式识别为核心主线，核心目标是从传感器阵列响应中提取 “气味指纹” 以完成气味分类、定量或未知气味分组，整体可分为传统统计、机器学习与深度学习三大类。
2.首先通过去噪、基线校正、归一化等信号预处理步骤消除传感器漂移与环境干扰，再提取稳态响应、峰值等特征并借助 PCA、LDA 等方法降维去冗余，最后通过模式识别实现核心任务；
3.其中传统算法以 KNN、SVM、PLSR、K-Means 等为主，适合小样本高维数据，深度学习算法如 CNN、LSTM、Transformer 等则能捕捉时空特征，实现端到端识别，是当前主流趋势。
4.目前电子鼻算法仍面临传感器漂移、低选择性、小样本泛化等挑战，优化方向集中在自适应校正、迁移学习、轻量化模型等，广泛应用于食品检测、环境监测、医疗诊断等领域，未来将向时空融合、多模态融合、材料 - 算法协同优化方向发展。'''

# 构造评估数据集所需数据
questions = [response['input']]          # 测试问题
answers = [response['answer']]           # 模型回答
contexts = [[doc.page_content for doc in response['context']]]  # 检索到的上下文
ground_truth = [truth]                   # 真实答案

# 构建评估数据集字典
data_samples = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truth
}

# 转换为RAGAS兼容的Dataset格式
dataset = Dataset.from_dict(data_samples)

# 获取大模型和嵌入模型实例
llms = get_lc_model_client()
embedding = get_ali_embeddings()

# 封装模型为RAGAS兼容接口（适配LangChain模型调用规范）
vllm = LangchainLLMWrapper(llms)
vllm_e = LangchainEmbeddingsWrapper(embedding)

# 执行RAGAS评估（核心指标：上下文精准度、召回率、答案忠实度、答案相关性）
result = evaluate(
    dataset,
    llm=vllm,
    embeddings=vllm_e,
    metrics=[
        context_precision,  # 上下文精准度：检索的上下文是否都和问题相关
        context_recall,     # 上下文召回率：是否检索到所有回答所需的关键信息
        faithfulness,       # 答案忠实度：回答是否完全基于检索的上下文（无幻觉）
        answer_relevancy,   # 答案相关性：回答是否紧密围绕问题（无无关内容）
    ],
)
print(result)