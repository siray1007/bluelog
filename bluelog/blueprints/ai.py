# -*- coding: utf-8 -*-
"""
    :author: Grey Li (李辉)
    :url: http://greyli.com
    :copyright: © 2018 Grey Li <withlihui@gmail.com>
    :license: MIT, see LICENSE for more details.
"""
from flask import render_template, request, current_app, Blueprint, jsonify
from openai import OpenAI
from typing import List, Dict, Any
import openai
import os


ai_bp = Blueprint('ai', __name__)


class AIClient:
    """AI客户端类,用于与AI模型进行交互"""
    def __init__(self):
        # 延迟初始化，在实际调用时获取配置
        self.client = None
        self.model = None

    def _initialize_client(self):
        """初始化AI客户端"""
        if self.client is None:
            # 从配置获取API密钥
            api_key = current_app.config.get('AI_API_KEY')
            base_url = current_app.config.get('AI_BASE_URL')
            model = current_app.config.get('AI_MODEL')

            # 记录环境变量和配置值用于调试
            current_app.logger.debug(f"AI_API_KEY from config: {'*' * len(api_key) if api_key else 'None'}")
            current_app.logger.debug(f"AI_BASE_URL from config: {base_url}")
            current_app.logger.debug(f"AI_MODEL from config: {model}")
            env_api_key = os.environ.get('AI_API_KEY', '')
            masked_env_key = '*' * len(env_api_key) if env_api_key else 'None'
            current_app.logger.debug(f"AI_API_KEY from env: {masked_env_key}")

            if not api_key:
                raise Exception("AI_API_KEY is not configured. Please check your environment variables.")

            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            self.model = model

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
        # 确保客户端已初始化
        self._initialize_client()

        return self._make_api_call(messages)

    def _make_api_call(self, messages):
        """执行API调用"""
        try:
            current_app.logger.debug(f"Sending request to AI model with messages: {messages}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=500,  # 增加生成的token数量
                temperature=0.7,  # 调整生成文本的多样性
                timeout=60  # 设置超时时间
            )
            current_app.logger.debug("Successfully sent request to AI model")
            return response
        except Exception as e:
            self._handle_api_exception(e)

    def _handle_api_exception(self, e: Exception):
        """
        统一处理API异常

        Args:
            e: 异常对象
        """
        # 定义错误类型映射
        error_mapping = {
            openai.APIConnectionError: "API连接错误",
            openai.AuthenticationError: "认证错误",
            openai.PermissionDeniedError: "权限不足",
            openai.RateLimitError: "请求过于频繁",
            openai.InternalServerError: "内部错误"
        }

        # 查找匹配的错误类型
        error_type = "API调用失败"
        for error_class, mapped_type in error_mapping.items():
            if isinstance(e, error_class):
                error_type = mapped_type
                break

        # 处理OpenAI相关错误和其他异常
        if isinstance(e, openai.OpenAIError):
            error_msg = f"OpenAI API {error_type}: {str(e)}"
            current_app.logger.error(error_msg)
            raise Exception(f"AI服务{error_type}: {str(e)}")
        else:
            error_msg = f"获取流式模型响应失败: {str(e)}"
            current_app.logger.error(error_msg)
            raise Exception(error_msg)

    def _handle_api_error(self, error_type: str, e: Exception):
        """
        处理API相关错误

        Args:
            error_type: 错误类型描述
            e: 异常对象
        """
        error_msg = f"OpenAI API {error_type}: {str(e)}"
        current_app.logger.error(error_msg)
        raise Exception(f"AI服务{error_type}: {str(e)}")

    def _handle_general_error(self, e: Exception):
        """
        处理一般性错误

        Args:
            e: 异常对象
        """
        error_msg = f"获取流式模型响应失败: {str(e)}"
        current_app.logger.error(error_msg)
        raise Exception(error_msg)


@ai_bp.route('/')
def index():
    return render_template('ai/index.html')


@ai_bp.route('/chat', methods=['POST'])
def chat():
    try:
        current_app.logger.debug("Chat endpoint called")
        data = request.get_json()
        current_app.logger.debug(f"Received data: {data}")

        if not data:
            current_app.logger.error("No data received in request")
            return jsonify({'error': '请求数据不能为空'}), 400

        user_message = data.get('message', '')
        history = data.get('history', [])

        # 验证用户输入
        if not validate_user_message(user_message):
            current_app.logger.debug("User message validation failed")
            return jsonify({'error': '消息不能为空'}), 400

        # 添加用户消息到历史记录
        history.append({"role": "user", "content": user_message})
        current_app.logger.debug(f"Updated history: {history}")

        # 获取AI响应
        response_text = get_ai_response(history)
        current_app.logger.debug(f"AI response: {response_text}")

        # 添加助手回复到历史记录
        history.append({"role": "assistant", "content": response_text})

        return jsonify({'response': response_text, 'history': history})
    except openai.OpenAIError as e:  # 捕获 OpenAI 相关的异常
        current_app.logger.error(f'OpenAI API 调用失败: {str(e)}')
        return jsonify({'error': f'OpenAI API 调用失败: {str(e)}'}), 500
    except Exception as e:  # 捕获其他异常
        current_app.logger.error(f'AI服务出错: {str(e)}')
        import traceback
        current_app.logger.error(traceback.format_exc())  # 记录完整的堆栈信息
        return jsonify({'error': f'AI服务出错: {str(e)}'}), 500


def validate_user_message(user_message):
    """验证用户消息是否有效"""
    return bool(user_message and user_message.strip())


def get_ai_response(history):
    """获取AI响应"""
    try:
        current_app.logger.debug("Initializing AI client in get_ai_response")
        # 初始化AI客户端
        ai_client = AIClient()

        # 获取AI响应
        stream = ai_client.get_completion_stream(history)

        # 处理流式响应
        response_text = ""
        for chunk in stream:
            if (not hasattr(chunk, 'choices') or not chunk.choices or
                    not hasattr(chunk.choices[0], 'delta') or
                    not hasattr(chunk.choices[0].delta, 'content')):
                current_app.logger.error(f"Invalid response format from AI model: {chunk}")
                raise Exception("Invalid response format from AI model")
            if chunk.choices and chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content

        return response_text
    except Exception as e:
        current_app.logger.error(f"Error in get_ai_response: {str(e)}")
        raise
