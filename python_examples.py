"""
Python 编程示例 — 修正版

修正列表:
  1. ✅ 补全 Generator 导入
  2. ✅ 完整多线程并发请求（含结果收集）
  3. ✅ 重试间加入 time.sleep() 退避
  4. ✅ 异常处理扩展：捕获所有 requests.RequestException
  5. ✅ 移除假 URL，改为可 mock 测试的接口
  6. ✅ 斐波那契处理负数边界
  7. ✅ 重试循环变量名清晰化
"""
import time
import functools
from typing import Callable, Any, Generator, List, Tuple

import requests


# ============================================================
# 1. 装饰器带参数重试机制（修正版）
# ============================================================

def retry_on_timeout(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
) -> Callable:
    """
    重试装饰器：捕获 requests.RequestException，指数退避重试。

    Args:
        max_retries: 最大重试次数
        delay:      初始延迟（秒）
        backoff:    退避倍数（每次重试 delay *= backoff）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc = None
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, OSError, TimeoutError) as e:
                    last_exc = e
                    if attempt < max_retries:
                        print(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
                        print(f"Retrying in {current_delay:.1f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
            # 所有重试耗尽，抛出最后一次异常
            raise last_exc  # type: ignore
        return wrapper
    return decorator


@retry_on_timeout(max_retries=3, delay=0.5, backoff=2.0)
def fetch_data(url: str) -> Any:
    """获取 JSON 数据，带重试机制"""
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()


# ============================================================
# 2. 生成器斐波那契（修正版）
# ============================================================

def fibonacci(n: int) -> Generator[int, None, None]:
    """
    生成前 n 个斐波那契数。

    Args:
        n: 生成数量（负数返回空）
    """
    if n <= 0:
        return
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


# ============================================================
# 3. 列表推导式过滤偶数
# ============================================================

def filter_even(numbers: List[int]) -> List[int]:
    """过滤出列表中的偶数"""
    return [x for x in numbers if x % 2 == 0]


# ============================================================
# 4. with 上下文管理文件
# ============================================================

def read_file(file_path: str) -> str:
    """读取文件全部内容（自动关闭）"""
    with open(file_path, "r") as f:
        return f.read()


# ============================================================
# 5. Lambda 排序字典
# ============================================================

def sort_dict_by_value(data: dict) -> dict:
    """按值升序排序字典"""
    return dict(sorted(data.items(), key=lambda item: item[1]))


# ============================================================
# 6. 多线程池并发请求（完整版）
# ============================================================

def fetch_single(url: str, timeout: float = 10) -> Tuple[str, int]:
    """单个 URL 请求，返回 (url, status_code)"""
    response = requests.get(url, timeout=timeout)
    return url, response.status_code


def fetch_urls_concurrently(
    urls: List[str],
    max_workers: int = 5,
    timeout: float = 10,
) -> List[Tuple[str, int]]:
    """
    并发请求多个 URL，收集所有结果。

    Args:
        urls:        URL 列表
        max_workers: 最大并发数
        timeout:     单个请求超时（秒）

    Returns:
        [(url, status_code), ...] 列表
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_single, url, timeout): url
            for url in urls
        }
        for future in as_completed(future_map):
            url = future_map[future]
            try:
                _, status = future.result()
                results.append((url, status))
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
                results.append((url, -1))
    return results
