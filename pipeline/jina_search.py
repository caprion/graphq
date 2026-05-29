#!/usr/bin/env python3
"""
Jina Reader + Multi-Tier Search Integration
==========================================
- Jina Reader (r.jina.ai) for URL → markdown conversion
- Tavily search as primary, Brave Search fallback, DuckDuckGo last resort

Usage:
    from jina_search import fetch_url, search_web
    markdown = fetch_url("https://example.com/article")
    results = search_web("query", provider="tavily")
"""

import urllib.request
import urllib.parse
import json, re, time

# =============================================================================
# JINA READER — URL to Markdown
# =============================================================================

def fetch_url(url, timeout=15):
    """
    Fetch a URL and convert to markdown using Jina Reader (r.jina.ai).
    
    Jina Reader strips ads, navigation, and returns clean content.
    Free, no API key needed for basic usage.
    
    Args:
        url: the URL to fetch
        timeout: request timeout in seconds
    
    Returns:
        markdown string, or None if failed
    """
    # Jina Reader endpoint: just prepend https://r.jina.ai/
    jina_url = f"https://r.jina.ai/{url}"
    
    headers = {
        "Accept": "text/markdown",
        "X-Return-Format": "markdown",
        "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)"
    }
    
    try:
        req = urllib.request.Request(jina_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = resp.read().decode("utf-8", errors="replace").strip()
                
                # Jina returns JSON with structure: {"code":200,"status":20000,"data":{"content":"..."}}
                # or raw markdown if X-Return-Format is set
                
                # Try JSON parse first (newer Jina format)
                try:
                    data = json.loads(body)
                    if isinstance(data, dict):
                        # Check for new format: data.content
                        if "data" in data and isinstance(data["data"], dict):
                            content = data["data"].get("content", "")
                            if content:
                                return content
                        # Fallback: look for content at top level
                        if "content" in data:
                            return data["content"]
                        # Another possible format: data.data.content (nested)
                        if "data" in data and isinstance(data["data"], dict):
                            if "content" in data["data"]:
                                return data["data"]["content"]
                except json.JSONDecodeError:
                    pass
                
                # If not JSON, treat as raw markdown
                if body:
                    return body
                    
            return None
    except Exception as e:
        print(f"  ⚠ Jina fetch failed for {url}: {e}")
        return None


# =============================================================================
# MULTI-PROVIDER SEARCH (Tavily → Brave → DuckDuckGo)
# =============================================================================

def search_tavily(query, api_key=None, max_results=10):
    """
    Search using Tavily API.
    
    Args:
        query: search query
        api_key: Tavily API key (from vault: TAVILY_API_KEY)
        max_results: max results to return
    
    Returns:
        list of {"url": ..., "title": ..., "description": ..., "content": ...}
    """
    # Try vault first
    if not api_key:
        api_key = get_vault_secret("TAVILY_API_KEY")
    
    if not api_key:
        return None  # Signal to use fallback
    
    try:
        import urllib.request
        
        payload = json.dumps({
            "query": query,
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False
        }).encode("utf-8")
        
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key
            },
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            return [
                {
                    "url": r["url"],
                    "title": r.get("title", ""),
                    "description": r.get("description", ""),
                    "content": r.get("content", "")
                }
                for r in results
            ]
    except Exception as e:
        print(f"  ⚠ Tavily failed: {e}")
        return None  # Fallback to Brave


def search_brave(query, api_key=None, max_results=10):
    """
    Search using Brave Search API (free tier: 2000 queries/month).
    
    Get key: https://api.search.brave.com/auth/keys
    
    Args:
        query: search query
        api_key: Brave API key
        max_results: max results
    
    Returns:
        list of search results
    """
    if not api_key:
        api_key = get_vault_secret("BRAVE_SEARCH_API_KEY")
    
    if not api_key:
        return None
    
    try:
        import urllib.request
        
        query_encoded = urllib.parse.quote(query)
        url = f"https://api.search.brave.com/res/v1/search?q={query_encoded}&count={max_results}"
        
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
            "User-Agent": "Mozilla/5.0 HermesAgent/1.0"
        })
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("web", {}).get("results", [])
            return [
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "description": r.get("description", ""),
                    "content": r.get("description", "")  # Brave gives description as snippet
                }
                for r in results
            ]
    except Exception as e:
        print(f"  ⚠ Brave failed: {e}")
        return None  # Fallback to DuckDuckGo


def search_duckduckgo(query, max_results=10):
    """
    DuckDuckGo HTML scrape (fallback, last resort).
    Uses thelitecoin/ddg-search approach or direct HTML parsing.
    
    Note: DDG blocks automated scraping. Use sparingly.
    
    Returns:
        list of search results
    """
    try:
        import urllib.request
        
        query_encoded = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={query_encoded}"
        
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
            
        # Parse results from DDG HTML
        results = []
        # DDG result pattern: <a class="result__a" href="...">title</a>
        # and <a class="result__snippet" href="...">description</a>
        
        import re
        link_pattern = re.compile(r'<a class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL)
        snippet_pattern = re.compile(r'<a class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
        
        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)
        
        for i, (url, title) in enumerate(links[:max_results]):
            title = re.sub(r'<[^>]+>', '', title).strip()
            desc = re.sub(r'<[^>]+>', '', snippets[i]) if i < len(snippets) else ""
            results.append({
                "url": url,
                "title": title,
                "description": desc,
                "content": desc
            })
        
        return results[:max_results]
        
    except Exception as e:
        print(f"  ⚠ DuckDuckGo failed: {e}")
        return []


def search_web(query, provider_priority=["tavily", "brave", "duckduckgo"], 
                tavily_key=None, brave_key=None, max_results=10):
    """
    Multi-tier search with automatic fallback.
    
    Tries providers in order until one succeeds.
    
    Args:
        query: search query string
        provider_priority: list of providers to try in order
        tavily_key: Tavily API key
        brave_key: Brave Search API key  
        max_results: max results per provider
    
    Returns:
        list of search results (each with url, title, description, content)
    """
    results = None
    tried = []
    
    for provider in provider_priority:
        if provider == "tavily":
            tried.append("Tavily")
            results = search_tavily(query, api_key=tavily_key, max_results=max_results)
        elif provider == "brave":
            tried.append("Brave")
            results = search_brave(query, api_key=brave_key, max_results=max_results)
        elif provider == "duckduckgo":
            tried.append("DuckDuckGo")
            results = search_duckduckgo(query, max_results=max_results)
        
        if results is not None and len(results) > 0:
            print(f"  ✅ {provider.upper()} returned {len(results)} results")
            return results
        elif results is not None:
            print(f"  ⚠ {provider.upper()} returned empty — trying next...")
    
    print(f"  ❌ All search providers failed: {', '.join(tried)}")
    return []


# =============================================================================
# VAULT HELPER (Azure KeyVault via managed identity)
# =============================================================================

def get_vault_secret(secret_name, vault_name="claw-agent"):
    """
    Fetch a secret from Azure KeyVault using managed identity.
    
    Requires:
        - VM with managed identity enabled
        - Secret name in vault `claw-agent`
        - Read permission on the vault
    
    Args:
        secret_name: name of the secret (e.g., "TAVILY_API_KEY")
        vault_name: Azure KeyVault name
    
    Returns:
        secret value as string, or None if not found/inaccessible
    """
    try:
        import urllib.request
        
        identity_endpoint = "http://169.254.169.254/metadata/identity/oauth2/token"
        token_url = f"{identity_endpoint}?api-version=2023-11-01&resource=https%3A%2F%2Fvault.azure.net"
        
        req = urllib.request.Request(token_url, headers={"Metadata": "true"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            token_data = json.loads(resp.read().decode("utf-8"))
            access_token = token_data["access_token"]
        
        # Fetch secret
        secret_url = f"https://{vault_name}.vault.azure.net/secrets/{secret_name}/?api-version=2023-07-01"
        secret_req = urllib.request.Request(secret_url, headers={
            "Authorization": f"Bearer {access_token}"
        })
        
        with urllib.request.urlopen(secret_req, timeout=10) as resp:
            secret_data = json.loads(resp.read().decode("utf-8"))
            return secret_data.get("value")
            
    except Exception as e:
        print(f"  ⚠ Vault fetch failed for {secret_name}: {e}")
        return None


# =============================================================================
# INTEGRATED PIPELINE — JINA + NPMI-GEE-LIDER
# =============================================================================

def build_from_weburls(urls, npmi_window=5, alpha=0.5):
    """
    Fetch URLs via Jina, build NPMI+GEE+LIDER pipeline from the content.
    
    Args:
        urls: list of URLs to fetch
        npmi_window: NPMI co-occurrence window size
        alpha: GEE normalization exponent
    
    Returns:
        NPMGEEPipeline with content from all URLs
    """
    from npmi_gee_lider import NPMGEEPipeline
    
    print(f"\n🌐 Fetching {len(urls)} URLs via Jina Reader...")
    
    documents = []
    successful_urls = []
    
    for url in urls:
        print(f"  → Fetching: {url[:60]}...")
        content = fetch_url(url)
        if content and len(content.strip()) > 100:
            # Truncate to first 5000 chars for reasonable processing
            doc_text = content[:5000]
            documents.append(doc_text)
            successful_urls.append(url)
            print(f"    ✅ Got {len(doc_text)} chars")
        else:
            print(f"    ❌ Failed or too short")
    
    if not documents:
        print("  ❌ No content fetched from any URL")
        return None
    
    print(f"\n📦 Building pipeline from {len(documents)} fetched documents...")
    
    pipeline = NPMGEEPipeline(k_classes=None, npmi_window=npmi_window, alpha=alpha)
    pipeline.build_from_text_corpus(documents)
    
    return pipeline


# =============================================================================
# DEMO
# =============================================================================

def demo_search():
    """Demo multi-tier search"""
    print("\n" + "="*70)
    print("Multi-Tier Search Demo")
    print("="*70)
    
    queries = [
        "what is Graph Encoder Embedding GEE",
        "LiteParse PDF parser"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = search_web(query, provider_priority=["tavily", "brave", "duckduckgo"])
        for r in results[:3]:
            print(f"  • {r['title'][:60]}")
            print(f"    {r['url'][:70]}")


def demo_jina():
    """Demo Jina URL fetching"""
    print("\n" + "="*70)
    print("Jina Reader Demo")
    print("="*70)
    
    urls = [
        "https://en.wikipedia.org/wiki/Graph_theory",
        "https://github.com/run-llama/liteparse"
    ]
    
    for url in urls:
        print(f"\n📄 Fetching: {url[:60]}...")
        content = fetch_url(url)
        if content:
            preview = content[:200].replace('\n', ' ')
            print(f"  ✅ Preview: {preview}...")
        else:
            print(f"  ❌ Failed")


if __name__ == "__main__":
    demo_jina()
    demo_search()