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
            api_key = current_app.config.get('AI_API_KEY', 'sk-0b41758d30e0441d9a90a69c74cbbb35')

            self.client = OpenAI(
                api_key=api_key,
                base_url=current_app.config.get('AI_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
            )
            self.model = current_app.config.get('AI_MODEL', 'deepseek-r1-0528')

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

        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=50,  # 减少生成的token数量
                temperature=0.7,  # 调整生成文本的多样性
                timeout=60  # 设置超时时间
                )
        except Exception as e:
            raise Exception(f"获取流式模型响应失败: {str(e)}")


@ai_bp.route('/')
def index():
    return render_template('ai/index.html')


@ai_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    history = data.get('history', [])

    if not user_message:
        return jsonify({'error': '消息不能为空'}), 400

    # 添加用户消息到历史记录
    history.append({"role": "user", "content": user_message})

    try:
        # 初始化AI客户端
        ai_client = AIClient()

        # 获取AI响应
        stream = ai_client.get_completion_stream(history)

        # 处理流式响应
        response_text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content

        # 添加助手回复到历史记录
        history.append({"role": "assistant", "content": response_text})

        return jsonify({'response': response_text, 'history': history})

    except Exception as e:
        return jsonify({'error': f'AI服务出错: {str(e)}'}), 500
