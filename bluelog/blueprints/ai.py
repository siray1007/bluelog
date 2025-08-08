# -*- coding: utf-8 -*-
"""
    :author: Grey Li (李辉)
    :url: http://greyli.com
    :copyright: © 2018 Grey Li <withlihui@gmail.com>
    :license: MIT, see LICENSE for more details.
"""
from flask import render_template, request, current_app, Blueprint, jsonify
from flask import Response, stream_with_context
import json
from openai import OpenAI
from typing import List, Dict, Any
import openai
import os
import traceback
import sys
import datetime

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

            # 清除任何可能的代理配置残留
            for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                if env_var in os.environ:
                    current_app.logger.warning(f"环境变量中存在代理配置 {env_var}，可能影响连接")

            # 打印配置（脱敏API_KEY）
            current_app.logger.debug(
                f"Loaded AI config - API_KEY: {'*'*len(api_key) if api_key else 'None'}, "
                f"BASE_URL: {base_url}, MODEL: {model}"
            )

            # 记录环境变量和配置值用于调试
            current_app.logger.debug(f"AI_API_KEY from config: {'*' * len(api_key) if api_key else 'None'}")
            current_app.logger.debug(f"AI_BASE_URL from config: {base_url}")
            current_app.logger.debug(f"AI_MODEL from config: {model}")

            env_api_key = os.environ.get('AI_API_KEY', '')
            masked_env_key = '*' * len(env_api_key) if env_api_key else 'None'
            current_app.logger.debug(f"AI_API_KEY from env: {masked_env_key}")

        # 检查必要配置 - 修复变量作用域问题
        api_key = current_app.config.get('AI_API_KEY')
        base_url = current_app.config.get('AI_BASE_URL')
        model = current_app.config.get('AI_MODEL')

        if not api_key:
            current_app.logger.error("AI_API_KEY is empty in config and environment")
            raise Exception("AI_API_KEY is not configured. Please check your environment variables.")

        if not base_url:
            current_app.logger.error("AI_BASE_URL is not configured")
            raise Exception("AI_BASE_URL is not configured")

        if not model:
            current_app.logger.error("AI_MODEL is not configured")
            raise Exception("AI_MODEL is not configured")

        # 初始化客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

        current_app.logger.debug(f"Calling model {self.model} at {self.client.base_url}")

    def get_completion_stream(self, messages: List[Dict[str, str]]) -> Any:
        """获取模型流式完成响应"""
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
                max_tokens=500,
                temperature=0.7,
                timeout=60
            )
            current_app.logger.debug("Successfully sent request to AI model")
            return response
        except Exception as e:
            self._handle_api_exception(e)

    def _handle_api_exception(self, e: Exception):
        """统一处理API异常"""
        error_mapping = {
            openai.APIConnectionError: "API连接错误",
            openai.AuthenticationError: "认证错误",
            openai.PermissionDeniedError: "权限不足",
            openai.RateLimitError: "请求过于频繁",
            openai.InternalServerError: "内部错误"
        }

        error_type = "API调用失败"
        for error_class, mapped_type in error_mapping.items():
            if isinstance(e, error_class):
                error_type = mapped_type
                break

        if isinstance(e, openai.OpenAIError):
            error_msg = f"OpenAI API {error_type}: {str(e)}"
            current_app.logger.error(error_msg)
            raise Exception(f"AI服务{error_type}: {str(e)}")
        else:
            error_msg = f"获取流式模型响应失败: {str(e)}"
            current_app.logger.error(error_msg)
            raise Exception(error_msg)


def _extract_and_validate_request_data():
    """提取并验证请求数据"""
    data = request.get_json()
    current_app.logger.debug(f"Received data: {data}")

    if not data:
        current_app.logger.error("No data received in request")
        return None, jsonify({'error': '请求数据不能为空'}), 400

    user_message = data.get('message', '')
    history = data.get('history', [])

    if not validate_user_message(user_message):
        current_app.logger.debug("User message validation failed")
        return None, jsonify({'error': '消息不能为空'}), 400

    # 添加用户消息到历史记录
    history.append({"role": "user", "content": user_message})
    current_app.logger.debug(f"Updated history: {history}")

    return (user_message, history), None, None


def _create_stream_generator(history):
    """创建流式响应生成器"""
    def generate():
        ai_client = AIClient()
        stream = ai_client.get_completion_stream(history)

        response_text = ""
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content is not None:
                    content = delta.content  # 保留原始格式
                    response_text += content
                    yield f"data: {json.dumps({'response': content, 'done': False})}\n\n"

        # 发送完成标志和完整历史
        history.append({"role": "assistant", "content": response_text})
        yield f"data: {json.dumps({'response': '', 'done': True, 'history': history})}\n\n"

    return generate


@ai_bp.route('/')
def index():
    return render_template('ai/index.html')


@ai_bp.route('/chat', methods=['POST'])
def chat():
    """处理聊天请求的主函数"""
    try:
        current_app.logger.debug("Chat endpoint called")

        # 提取和验证请求数据
        data_result, error_response, status_code = _extract_and_validate_request_data()
        if error_response:
            return error_response, status_code

        _, history = data_result

        # 创建流式响应
        generate = _create_stream_generator(history)
        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    except openai.OpenAIError as e:
        current_app.logger.error(f'OpenAI API 调用失败: {str(e)}')
        return jsonify({'error': f'OpenAI API 调用失败: {str(e)}'}), 500
    except Exception as e:
        print("=== AI 聊天接口错误 ===", file=sys.stderr)
        print(f"时间: {datetime.datetime.now()}", file=sys.stderr)
        print(f"错误信息: {str(e)}", file=sys.stderr)
        print("堆栈跟踪:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify({'error': f'AI服务出错: {str(e)}'}), 500


def validate_user_message(user_message):
    """验证用户消息是否有效"""
    return bool(user_message and user_message.strip())


def get_ai_response(history):
    """获取AI响应"""
    try:
        current_app.logger.debug("Initializing AI client in get_ai_response")
        ai_client = AIClient()
        stream = ai_client.get_completion_stream(history)

        response_text = ""
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content is not None:
                    response_text += delta.content.strip()
            else:
                current_app.logger.debug(f"Ignoring empty chunk: {chunk}")

        if not response_text.strip():
            raise Exception("AI returned empty response. Please try again.")

        return response_text
    except Exception as e:
        current_app.logger.error(f"Error in get_ai_response: {str(e)}")
        raise
