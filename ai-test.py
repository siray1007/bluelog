import threading
import time
from openai import OpenAI


def qwen_plus_test():
    """
    Qwen-Plus模型测试函数
    """
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"},
        ],
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
    )
    print(completion.model_dump_json())


def show_waiting_message(stop_event, start_time):
    """
    显示等待消息

    Args:
        stop_event: 用于控制线程停止的事件
        start_time: 开始时间
    """
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        print(f"\r助手: 请等待 (已等待: {elapsed:.2f}秒)", end='', flush=True)
        time.sleep(0.1)  # 每0.1秒更新一次显示


def call_model_api(messages):
    """
    调用模型API

    Args:
        messages: 对话历史消息

    Returns:
        模型响应结果
    """
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
    )
    return completion


def handle_api_exception(e, stop_event=None, wait_thread=None):
    """
    处理API调用异常

    Args:
        e: 异常对象
        stop_event: 等待线程控制事件
        wait_thread: 等待线程
    """
    # 停止等待时间显示线程（如果正在运行）
    if stop_event:
        stop_event.set()
        if wait_thread:
            wait_thread.join()

    # 清除等待提示并回到行首
    print("\r" + " " * 50 + "\r", end='')
    print(f"错误: {e}")


def terminal_chat():
    """
    终端问答程序
    """
    print("欢迎使用终端问答程序！输入 'quit' 或 'exit' 退出程序。")
    print("-" * 50)

    # 初始化对话历史
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    while True:
        # 获取用户输入
        user_input = input("\n请输入问题: ").strip()

        # 检查退出条件
        if user_input.lower() in ['quit', 'exit']:
            print("助手: 再见！")
            break

        if not user_input:
            continue

        # 添加用户消息到历史
        messages.append({"role": "user", "content": user_input})

        try:
            # 用于控制等待时间显示的线程
            stop_event = threading.Event()
            start_time = time.time()

            # 启动等待时间显示线程
            wait_thread = threading.Thread(target=show_waiting_message, args=(stop_event, start_time))
            wait_thread.start()

            # 调用模型API
            completion = call_model_api(messages)

            # 停止等待时间显示线程
            stop_event.set()
            wait_thread.join()

            # 计算总等待时间
            elapsed_time = time.time() - start_time

            # 清除等待提示并回到行首
            print("\r" + " " * 50 + "\r", end='')

            # 获取并打印助手回复
            assistant_reply = completion.choices[0].message.content
            print(f"助手: {assistant_reply}")
            print(f"     (响应时间: {elapsed_time:.2f}秒)")

            # 添加助手回复到历史
            messages.append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            handle_api_exception(e, stop_event, wait_thread)


if __name__ == "__main__":
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-0b41758d30e0441d9a90a69c74cbbb35",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 根据需要调用不同函数
    # qwen_plus_test()
    # test_image()
    terminal_chat()
