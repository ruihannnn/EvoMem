"""
测试项目中的搜索工具包
使用项目真实的 API 调用流程
"""
import asyncio
import os

from dotenv import find_dotenv, load_dotenv

# 加载.env文件
load_dotenv(find_dotenv(raise_error_if_not_found=False), verbose=True, override=True)


async def test_google_search():
    """测试 Google 搜索（通过 Serper API）"""
    print("\n" + "=" * 60)
    print("测试 1: Google Search (项目工具)")
    print("=" * 60)
    
    # 检查配置
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("❌ 错误: 请配置 SERPER_API_KEY")
        return False
    
    print(f"SERPER_API_KEY: {api_key[:10]}...")
    
    try:
        from utu.tools.search.google_search import GoogleSearch
        
        # 初始化搜索引擎
        search = GoogleSearch()
        
        # 执行搜索
        query = "Python programming"
        print(f"\n执行搜索: '{query}'")
        result = await search.search(query, num_results=3)
        
        print(f"\n✅ Google 搜索成功!")
        print(f"\n搜索结果:")
        print("-" * 60)
        print(result)
        print("-" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Google 搜索失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_jina_crawl():
    """测试 Jina 网页抓取"""
    print("\n" + "=" * 60)
    print("测试 2: Jina Crawl (项目工具)")
    print("=" * 60)
    
    # 检查配置
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        print("⚠️  跳过: 未配置 JINA_API_KEY")
        print("   可以从 https://jina.ai 获取免费 API Key")
        return None
    
    print(f"JINA_API_KEY: {api_key[:10]}...")
    
    try:
        from utu.tools.search.jina_crawl import JinaCrawl
        
        # 初始化爬虫
        crawler = JinaCrawl()
        
        # 抓取网页
        test_url = "https://www.python.org"
        print(f"\n抓取网页: {test_url}")
        result = await crawler.crawl(test_url)
        
        print(f"\n✅ Jina 抓取成功!")
        print(f"内容长度: {len(result)} 字符")
        print(f"\n内容预览:")
        print("-" * 60)
        print(result[:500] + "...")
        print("-" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Jina 抓取失败: {str(e)}")
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("   原因: API Key 无效或已过期")
            print("   解决: 从 https://jina.ai 获取新的 API Key")
        import traceback
        traceback.print_exc()
        return False


async def test_search_toolkit():
    """测试完整的 SearchToolkit（包括 search 和 web_qa）"""
    print("\n" + "=" * 60)
    print("测试 3: SearchToolkit - search() 方法")
    print("=" * 60)
    
    try:
        from utu.config import ToolkitConfig, ModelConfigs
        from utu.config.model_config import ModelProviderConfig, ModelParamsConfig
        from utu.tools.search_toolkit import SearchToolkit
        
        # 构建配置
        config = ToolkitConfig(
            name="search",
            config={
                "search_engine": "google",
                "search_params": {},
                "crawl_engine": "jina",
                "summary_token_limit": 10000,
            },
            config_llm=ModelConfigs(
                model_provider=ModelProviderConfig(
                    type="chat.completions",
                    model=os.getenv("UTU_LLM_MODEL"),
                    base_url=os.getenv("UTU_LLM_BASE_URL"),
                    api_key=os.getenv("UTU_LLM_API_KEY"),
                ),
                model_params=ModelParamsConfig()
            )
        )
        
        # 初始化工具包
        print("\n初始化 SearchToolkit...")
        toolkit = SearchToolkit(config)
        
        # 测试 search 方法
        query = "OpenAI GPT"
        print(f"\n执行搜索: '{query}'")
        result = await toolkit.search(query, num_results=3)
        
        print(f"\n✅ SearchToolkit.search() 成功!")
        print(f"\n搜索结果:")
        print("-" * 60)
        print(result)
        print("-" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ SearchToolkit 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_web_qa():
    """测试 SearchToolkit 的 web_qa 方法（需要 LLM 和 Jina API）"""
    print("\n" + "=" * 60)
    print("测试 4: SearchToolkit - web_qa() 方法")
    print("=" * 60)
    
    # 检查必要配置
    jina_key = os.getenv("JINA_API_KEY")
    llm_key = os.getenv("UTU_LLM_API_KEY")
    
    if not jina_key:
        print("⚠️  跳过: 未配置 JINA_API_KEY")
        return None
    
    if not llm_key:
        print("⚠️  跳过: 未配置 UTU_LLM_API_KEY (web_qa 需要 LLM 进行内容总结)")
        return None
    
    try:
        from utu.config import ToolkitConfig, ModelConfigs
        from utu.config.model_config import ModelProviderConfig, ModelParamsConfig
        from utu.tools.search_toolkit import SearchToolkit
        
        # 构建配置
        config = ToolkitConfig(
            name="search",
            config={
                "search_engine": "google",
                "search_params": {},
                "crawl_engine": "jina",
                "summary_token_limit": 10000,
            },
            config_llm=ModelConfigs(
                model_provider=ModelProviderConfig(
                    type="chat.completions",
                    model=os.getenv("UTU_LLM_MODEL"),
                    base_url=os.getenv("UTU_LLM_BASE_URL"),
                    api_key=os.getenv("UTU_LLM_API_KEY"),
                ),
                model_params=ModelParamsConfig(temperature=0.3)
            )
        )
        
        # 初始化工具包
        print("\n初始化 SearchToolkit (with LLM)...")
        toolkit = SearchToolkit(config)
        
        # 测试 web_qa 方法
        test_url = "https://www.python.org"
        query = "What is Python programming language?"
        print(f"\n执行 web_qa:")
        print(f"  URL: {test_url}")
        print(f"  Query: {query}")
        
        result = await toolkit.web_qa(test_url, query)
        
        print(f"\n✅ SearchToolkit.web_qa() 成功!")
        print(f"\nWeb QA 结果:")
        print("-" * 60)
        print(result)
        print("-" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ web_qa 测试失败: {str(e)}")
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("   原因: Jina API Key 无效或已过期")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("项目工具包 API 测试")
    print("=" * 60)
    
    # 显示代理配置
    http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
    https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
    if http_proxy or https_proxy:
        print(f"\n代理配置:")
        print(f"  http_proxy: {http_proxy}")
        print(f"  https_proxy: {https_proxy}")
    else:
        print("\n⚠️  未设置代理（如需访问外网，请设置代理）")
    
    results = []
    
    # 测试 1: Google 搜索
    result1 = await test_google_search()
    results.append(("Google Search", result1))
    
    # 测试 2: Jina 抓取
    result2 = await test_jina_crawl()
    results.append(("Jina Crawl", result2))
    
    # 测试 3: SearchToolkit.search()
    result3 = await test_search_toolkit()
    results.append(("SearchToolkit.search()", result3))
    
    # 测试 4: SearchToolkit.web_qa()
    result4 = await test_web_qa()
    results.append(("SearchToolkit.web_qa()", result4))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, result in results:
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "⚠️  跳过"
        print(f"{name:30s} {status}")
    
    print("\n" + "=" * 60)
    print("提示:")
    print("- Google Search 是 WebWalkerQA 的核心功能")
    print("- web_qa 需要 Jina API Key 和 LLM (用于内容总结)")
    print("- 如果 Jina API Key 无效，可切换到 crawl4ai")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
