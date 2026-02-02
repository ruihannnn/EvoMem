"""
API测试文件
测试OpenAI接口和Serper搜索API
"""

import json
import os

import requests
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
    
    # 从环境变量读取API Key和代理配置
    api_key = os.getenv("SERPER_API_KEY")
    http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
    https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
    
    print(f"API Key: {api_key[:10]}..." if api_key else "API Key: None")
    if http_proxy or https_proxy:
        print(f"代理配置: http_proxy={http_proxy}, https_proxy={https_proxy}")
    else:
        print("代理配置: 未设置（将直接连接）")
    
    if not api_key:
        print("❌ 错误: 请在.env文件中配置 SERPER_API_KEY")
        return
    
    try:
        # 准备请求数据
        payload = {
            "q": "apple inc"
        }
        
        # 设置请求头
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        # 配置代理（requests会自动使用环境变量中的代理，但也可以显式指定）
        proxies = {}
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        
        # 发送POST请求（requests会自动使用环境变量中的代理）
        response = requests.post(
            "https://google.serper.dev/search",
            json=payload,
            headers=headers,
            proxies=proxies if proxies else None,
            timeout=30
        )
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析并格式化输出
        result = response.json()
        print(f"\n✅ API调用成功!")
        print(f"HTTP状态码: {response.status_code}")
        print(f"搜索查询: apple inc")
        print(f"返回结果数: {len(result.get('organic', []))}")
        print(f"\n完整响应:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API调用失败: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text[:200]}")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")


def test_jina_api():
    """测试Jina Crawl API（直接测试）"""
    print("\n\n=== 测试 Jina Crawl API ===")
    
    # 从环境变量读取API Key和代理配置
    api_key = os.getenv("JINA_API_KEY")
    http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
    https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
    
    print(f"API Key: {api_key[:10]}..." if api_key else "API Key: None")
    if http_proxy or https_proxy:
        print(f"代理配置: http_proxy={http_proxy}, https_proxy={https_proxy}")
    else:
        print("代理配置: 未设置（将直接连接）")
    
    if not api_key:
        print("⚠️  警告: 未配置 JINA_API_KEY")
        print("   可以从 https://jina.ai 获取免费 API Key")
        return
    
    try:
        # 准备请求
        test_url = "https://www.python.org"
        url = f"https://r.jina.ai/{test_url}"
        headers = {
            'Authorization': f'Bearer {api_key}'
        }
        
        # 配置代理
        proxies = {}
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        
        print(f"\n抓取网页: {test_url}")
        response = requests.get(
            url,
            headers=headers,
            proxies=proxies if proxies else None,
            timeout=30
        )
        
        # 检查响应状态
        response.raise_for_status()
        
        print(f"\n✅ 网页抓取成功!")
        print(f"HTTP状态码: {response.status_code}")
        print(f"内容长度: {len(response.text)} 字符")
        print(f"\n内容预览:")
        print("-" * 60)
        print(response.text[:500] + "...")
        print("-" * 60)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 抓取失败: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text[:200]}")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")


def test_search_toolkit():
    """测试项目中的搜索工具包（需要 async）"""
    print("\n\n=== 测试 Search Toolkit (项目工具) ===")
    print("注意: 这是异步函数，需要使用 asyncio.run() 或 uv run 执行")
    print("请运行: uv run python -c 'import asyncio; from tests.test_toolkit import test_toolkit; asyncio.run(test_toolkit())'")
    print("或参考 tests/test-search-proxy.py 中的示例")


if __name__ == "__main__":
    print("=" * 60)
    print("API 测试套件")
    print("=" * 60)
    
    # 运行OpenAI API测试
    test_openai_api()
    
    # 运行Serper API测试
    test_serper_api()
    
    # 运行Jina API测试
    test_jina_api()
    
    # 提示异步工具测试
    test_search_toolkit()
    
    print("\n\n=== 测试完成 ===")
