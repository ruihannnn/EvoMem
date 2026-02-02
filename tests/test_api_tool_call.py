#!/usr/bin/env python3
"""测试 API 是否能正确识别工具调用"""

import asyncio
import json
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


async def test_tool_call():
    """测试工具调用"""
    from openai import AsyncOpenAI
    
    # 从环境变量读取配置
    api_key = os.getenv("UTU_LLM_API_KEY", "xxx")
    base_url = os.getenv("UTU_LLM_BASE_URL")
    model = os.getenv("UTU_LLM_MODEL", "qwen")
    llm_type = os.getenv("UTU_LLM_TYPE", "chat.completions")
    
    print("=" * 60)
    print("配置信息:")
    print(f"  LLM Type: {llm_type}")
    print(f"  Model: {model}")
    print(f"  Base URL: {base_url}")
    print(f"  API Key: {api_key[:20]}..." if api_key else "  API Key: None")
    print("=" * 60)
    
    # 创建客户端
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    # 定义一个简单的搜索工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web for information. Use this when you need to find current information or facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_qa",
                "description": "Extract specific answer from a webpage content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to extract information from"
                        },
                        "question": {
                            "type": "string",
                            "description": "The specific question to answer"
                        }
                    },
                    "required": ["url", "question"]
                }
            }
        }
    ]
    
    # 构造测试消息
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You MUST use the search tool to find information before answering questions."
        },
        {
            "role": "user",
            "content": "What is the capital of France? Please search for this information."
        }
    ]
    
    print("\n" + "=" * 60)
    print("发送请求...")
    print("=" * 60)
    print("\n系统提示:", messages[0]["content"])
    print("用户问题:", messages[1]["content"])
    print("\n可用工具:")
    for tool in tools:
        print(f"  - {tool['function']['name']}: {tool['function']['description']}")
    
    try:
        # 发送请求
        print("\n" + "=" * 60)
        print("调用 API...")
        print("=" * 60)
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.3,
            top_p=0.95
        )
        
        print("\n" + "=" * 60)
        print("API 响应:")
        print("=" * 60)
        
        # 打印完整响应
        print("\n完整响应 JSON:")
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        
        # 分析响应
        print("\n" + "=" * 60)
        print("响应分析:")
        print("=" * 60)
        
        choice = response.choices[0]
        message = choice.message
        
        print(f"\nFinish Reason: {choice.finish_reason}")
        
        if message.tool_calls:
            print(f"\n✅ 检测到工具调用! (共 {len(message.tool_calls)} 个)")
            for i, tool_call in enumerate(message.tool_calls, 1):
                print(f"\n工具调用 #{i}:")
                print(f"  ID: {tool_call.id}")
                print(f"  工具名: {tool_call.function.name}")
                print(f"  参数: {tool_call.function.arguments}")
        else:
            print("\n❌ 没有检测到工具调用!")
            
        if message.content:
            print(f"\n文本内容: {message.content}")
        
        # 打印 usage
        if response.usage:
            print(f"\nToken 使用情况:")
            print(f"  输入 tokens: {response.usage.prompt_tokens}")
            print(f"  输出 tokens: {response.usage.completion_tokens}")
            print(f"  总计 tokens: {response.usage.total_tokens}")
        
        return response
        
    except Exception as e:
        print(f"\n❌ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_without_tools():
    """测试不带工具的情况作为对照"""
    from openai import AsyncOpenAI
    
    api_key = os.getenv("UTU_LLM_API_KEY", "xxx")
    base_url = os.getenv("UTU_LLM_BASE_URL")
    model = os.getenv("UTU_LLM_MODEL", "qwen")
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
    
    print("\n\n" + "=" * 60)
    print("对照测试: 不带工具的情况")
    print("=" * 60)
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            top_p=0.95
        )
        
        print("\n响应内容:", response.choices[0].message.content)
        print("Finish Reason:", response.choices[0].finish_reason)
        
        return response
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return None


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("测试 1: 带工具的 API 调用")
    print("=" * 60)
    asyncio.run(test_tool_call())
    
    print("\n\n" + "=" * 60)
    print("测试 2: 不带工具的 API 调用 (对照)")
    print("=" * 60)
    asyncio.run(test_without_tools())
    
    print("\n\n测试完成!")
