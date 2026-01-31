"""
API测试文件
测试OpenAI接口和Serper搜索API
"""

import http.client
import json
import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

# 加载.env文件
load_dotenv(find_dotenv(raise_error_if_not_found=False), verbose=True, override=True)


def test_openai_api():
    """测试OpenAI接口"""
    print("\n=== 测试 OpenAI API ===")
    
    # 从环境变量读取配置
    model_name = os.getenv("UTU_LLM_MODEL")
    api_key = os.getenv("UTU_LLM_API_KEY")
    base_url = os.getenv("UTU_LLM_BASE_URL")
    
    print(f"模型名称: {model_name}")
    print(f"Base URL: {base_url}")
    print(f"API Key: {api_key[:10]}..." if api_key else "API Key: None")
    
    if not all([model_name, api_key, base_url]):
        print("❌ 错误: 请在.env文件中配置 UTU_LLM_MODEL, UTU_LLM_API_KEY 和 UTU_LLM_BASE_URL")
        return
    
    try:
        # 初始化OpenAI客户端
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 发送测试请求
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "hello"}
            ]
        )
        
        # 输出响应
        print(f"\n✅ API调用成功!")
        print(f"响应内容: {response.choices[0].message.content}")
        print(f"使用的模型: {response.model}")
        print(f"Token使用情况: {response.usage}")
        
    except Exception as e:
        print(f"❌ API调用失败: {str(e)}")


def test_serper_api():
    """测试Serper搜索API"""
    print("\n\n=== 测试 Serper Search API ===")
    
    # 从环境变量读取API Key
    api_key = os.getenv("SERPER_API_KEY")
    
    print(f"API Key: {api_key[:10]}..." if api_key else "API Key: None")
    
    if not api_key:
        print("❌ 错误: 请在.env文件中配置 SERPER_API_KEY")
        return
    
    try:
        # 建立HTTPS连接
        conn = http.client.HTTPSConnection("google.serper.dev")
        
        # 准备请求数据
        payload = json.dumps({
            "q": "apple inc"
        })
        
        # 设置请求头
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        # 发送POST请求
        conn.request("POST", "/search", payload, headers)
        
        # 获取响应
        res = conn.getresponse()
        data = res.read()
        
        # 解析并格式化输出
        result = json.loads(data.decode("utf-8"))
        print(f"\n✅ API调用成功!")
        print(f"搜索查询: apple inc")
        print(f"返回结果数: {len(result.get('organic', []))}")
        print(f"\n完整响应:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ API调用失败: {str(e)}")
    finally:
        conn.close()


if __name__ == "__main__":
    # 运行OpenAI API测试
    test_openai_api()
    
    # 运行Serper API测试
    test_serper_api()
    
    print("\n\n=== 测试完成 ===")
