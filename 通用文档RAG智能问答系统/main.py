import gradio as gr
from combine_client import CombineClient
from logger import setup_logger
from models import ALI_TONGYI_MAX_MODEL, ALI_TONGYI_DEEPSEEK_R1, ALI_TONGYI_DEEPSEEK_V3

# 定义可选的大模型列表
LLM_MODELS = [ALI_TONGYI_MAX_MODEL, ALI_TONGYI_DEEPSEEK_R1, ALI_TONGYI_DEEPSEEK_V3]

# 初始化知识库+大模型交互客户端
llm = CombineClient()
setup_logger()

def submit_show(query, chat_history):
    """
    处理用户提交：清空输入框，将用户问题添加到聊天记录
    """
    if query == '':
        return '', chat_history
    chat_history.append([query, None])
    return '', chat_history

def llm_reply(collection, chat_history, model, max_length=256, temperature=1):
    """
    生成模型回复（流式输出）：
    - 从聊天记录获取最新问题
    - 调用模型流式接口
    - 逐块拼接回复并更新聊天记录
    """
    question = chat_history[-1][0]
    # 调用流式回复接口
    response = llm.stream(question, collection, model=model, max_length=max_length, temperature=temperature)
    chat_history[-1][1] = ""

    # 逐块处理流式返回结果
    for chunk in response:
        if 'answer' in chunk:
            chunk_content = chunk['answer']
            if chunk_content is not None:
                chat_history[-1][1] += chunk_content
                yield chat_history

# 创建Gradio可视化界面
with gr.Blocks() as demo:
    # 页面标题
    gr.HTML("""<h1 align="center">通用文档分析助手(本程序正常运行需要科学上网)</h1>""")

    # 主布局：左侧聊天+模型配置，右侧参数+文件/知识库
    with gr.Row():
        with gr.Column(scale=4):
            # 模型选择下拉框
            model = gr.Dropdown(
                choices=LLM_MODELS,
                value=LLM_MODELS[0],
                label="LLM Model",
                interactive=True,
                scale=1
            )
            # 聊天机器人界面（支持复制）
            chatbot = gr.Chatbot(show_label=False, scale=3, show_copy_button=True)

        with gr.Column(scale=1) as column_config:
            # 模型参数配置
            max_length = gr.Slider(1, 8000, value=4000, step=100, label="模型回复最大长度", interactive=True)
            temperature = gr.Slider(0, 1.9, value=0.7, step=0.1, label="温度", interactive=True)
            # 功能按钮/控件
            clear = gr.Button("清除")
            collection = gr.Dropdown(label="知识库")  # 知识库选择下拉框
            file = gr.File(label="上传文件", file_types=['doc', 'docx', 'csv', 'txt', 'pdf', 'md'])  # 文件上传控件

    # 用户输入区域
    user_input = gr.Textbox(placeholder="Input...", show_label=False)
    user_submit = gr.Button("提交")

    # 绑定事件：清除按钮 → 清空聊天历史
    clear.click(fn=llm.clear_history, inputs=None, outputs=[chatbot])

    # 绑定事件：回车提交输入 → 先更新聊天记录 → 再生成回复
    user_input.submit(
        fn=submit_show,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    ).then(
        fn=llm_reply,
        inputs=[collection, chatbot, model, max_length, temperature],
        outputs=[chatbot]
    )

    # 绑定事件：点击提交按钮 → 逻辑同回车提交
    user_submit.click(
        fn=submit_show,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    ).then(
        fn=llm_reply,
        inputs=[collection, chatbot, model, max_length, temperature],
        outputs=[chatbot]
    )

    # 绑定事件：上传文件 → 处理文件并更新知识库列表
    file.upload(fn=llm.upload_knowledge, inputs=[file], outputs=[file, collection])

    # 绑定事件：切换知识库 → 清空聊天历史
    collection.change(fn=llm.clear_history, inputs=None, outputs=[chatbot])

    # 绑定事件：页面加载 → 清空聊天历史
    demo.load(fn=llm.clear_history, inputs=None, outputs=[chatbot])

# 启动Gradio应用
demo.launch()