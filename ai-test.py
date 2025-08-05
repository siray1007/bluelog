import threading
import time
from openai import OpenAI
from typing import List, Dict, Any, Optional


class AIClient:
    """AI客户端类,用于与AI模型进行交互"""
    def __init__(self):
        # 从环境变量获取API密钥，避免硬编码敏感信息
        api_key = "sk-0b41758d30e0441d9a90a69c74cbbb35"
        if not api_key:
            raise ValueError("未找到环境变量 DEEPSEEK_API_KEY,请先配置")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "deepseek-r1"

    def test_connection(self) -> None:
        """
        测试DeepSeek模型功能
        """
        print("正在测试模型连接...")
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "你好"}],
                timeout=30  # 设置超时时间
            )
            print("模型连接测试成功！")
            print(f"响应: {completion.choices[0].message.content}")
        except Exception as e:
            print(f"模型连接测试失败: {e}")

    def get_completion(self, messages: List[Dict[str, str]]) -> Any:
        """
        获取模型完成响应

        Args:
            messages: 对话历史消息

        Returns:
            模型响应结果

        Raises:
            Exception: 当API调用失败时抛出异常
        """
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=60  # 设置超时时间
            )
        except Exception as e:
            raise Exception(f"获取模型响应失败: {str(e)}")

    def get_completion_stream(self, messages: List[Dict[str, str]]) -> Any:
        """
        获取模型流式完成响应

        Args:
            messages: 对话历史消息

        Returns:
            模型流式响应结果

        Raises:
            Exception: 当API调用失败时抛出异常
        """
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                timeout=60  # 设置超时时间
            )
        except Exception as e:
            raise Exception(f"获取流式模型响应失败: {str(e)}")


class ChatApplication:
    """聊天应用程序类"""

    def __init__(self):
        """初始化聊天应用"""
        try:
            self.ai_client = AIClient()
        except ValueError as e:
            print(f"AI客户端初始化失败: {e}")
            exit(1)

        self.messages: List[Dict[str, str]] = []
        self._initialize_messages()

    def _initialize_messages(self) -> None:
        """初始化对话历史"""
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def show_welcome_message(self) -> None:
        """显示欢迎信息"""
        print("欢迎使用终端问答程序！")
        print("命令:")
        print("  'quit' 或 'exit' - 退出程序")
        print("  'history' - 查看对话历史")
        print("  'clear' - 清除对话历史")
        print("-" * 50)

    def show_waiting_message(self, stop_event: threading.Event, start_time: float) -> None:
        """
        显示等待消息

        Args:
            stop_event: 用于控制线程停止的事件
            start_time: 开始时间
        """
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            print(f"\r助手: 请等待 (耗时: {elapsed:.2f}秒)", end='', flush=True)
            time.sleep(0.1)

    def handle_user_command(self, user_input: str) -> tuple[bool, bool]:
        """
        处理用户命令

        Args:
            user_input: 用户输入

        Returns:
            (是否继续循环, 是否处理了命令)
        """
        user_input = user_input.lower().strip()

        # 检查退出条件
        if user_input in ['quit', 'exit']:
            print("助手: 再见！")
            return False, True

        # 显示对话历史
        if user_input == 'history':
            self._show_history()
            return True, True

        # 清除对话历史
        if user_input == 'clear':
            self._initialize_messages()
            print("对话历史已清除。")
            return True, True

        return True, False

    def _show_history(self) -> None:
        """显示对话历史"""
        print("\n对话历史:")
        print("-" * 30)
        for msg in self.messages:
            if msg["role"] == "system":
                continue  # 跳过系统消息
            print(f"{msg['role']}: {msg['content']}")
        print("-" * 30)

    def process_user_input(self, user_input: str) -> bool:
        """
        处理用户输入

        Args:
            user_input: 用户输入

        Returns:
            是否成功处理输入
        """
        if not user_input.strip():
            print("输入不能为空，请重新输入。")
            return False

        # 添加用户消息到历史
        self.messages.append({"role": "user", "content": user_input.strip()})
        return True

    def get_model_response(self) -> Optional[str]:
        """
        获取模型响应

        Returns:
            助手回复内容,如果出错返回None
        """
        start_time = time.time()  # 在开始处理前记录时间
        stop_event = threading.Event()
        wait_thread = None
        elapsed_time_at_api_call = 0

        try:
            # 启动等待时间显示线程
            wait_thread = threading.Thread(
                target=self.show_waiting_message,
                args=(stop_event, start_time),
                daemon=True  # 设置为守护线程，主程序退出时自动结束
            )
            wait_thread.start()

            # 调用模型API (使用流式)
            stream = self.ai_client.get_completion_stream(self.messages)

            # 记录API调用完成时的时间
            elapsed_time_at_api_call = time.time() - start_time

            # 停止等待时间显示线程
            stop_event.set()
            if wait_thread.is_alive():
                wait_thread.join(timeout=1)

            # 结束等待提示并在新行显示助手回复
            print()  # 换行，保留等待提示信息
            print("助手: ", end='', flush=True)

            # 处理流式响应
            assistant_reply = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    assistant_reply += content
                    print(content, end='', flush=True)

            print()  # 换行

            # 计算总耗时（使用API调用完成时的时间，保持一致性）
            print(f"     (总耗时: {elapsed_time_at_api_call:.2f}秒)")

            # 添加助手回复到历史
            self.messages.append({"role": "assistant", "content": assistant_reply})

            return assistant_reply

        except Exception as e:
            # 停止等待时间显示线程
            stop_event.set()
            if wait_thread and wait_thread.is_alive():
                wait_thread.join(timeout=1)

            # 结束等待提示并显示错误信息
            print()  # 换行，保留等待提示信息
            print(f"错误: {e}")
            return None

    def run(self) -> None:
        """运行聊天应用程序"""
        self.show_welcome_message()

        while True:
            try:
                # 获取用户输入
                user_input = input("\n请输入问题: ").strip()

                # 处理主循环逻辑
                if not self._process_main_loop(user_input):
                    break

            except KeyboardInterrupt:
                print("\n\n程序被用户中断，再见！")
                break
            except EOFError:
                # 处理输入流结束的情况（如Ctrl+D）
                print("\n\n输入流结束，程序退出。")
                break
            except Exception as e:
                print(f"发生未预期的错误: {e}")
                continue

    def _process_main_loop(self, user_input: str) -> bool:
        """
        处理主循环逻辑

        Args:
            user_input: 用户输入

        Returns:
            是否继续循环
        """
        # 处理用户命令
        should_continue, command_handled = self.handle_user_command(user_input)
        if not should_continue:
            return False

        if command_handled:
            return True

        # 处理用户输入
        if not self.process_user_input(user_input):
            return True

        # 获取模型响应
        self.get_model_response()
        return True


def main() -> None:
    """主函数"""
    try:
        # 创建并测试AI客户端
        ai_client = AIClient()
        ai_client.test_connection()
    except Exception as e:
        print(f"AI客户端测试失败: {e}")
        return

    # 运行聊天应用
    app = ChatApplication()
    app.run()


if __name__ == "__main__":
    main()
