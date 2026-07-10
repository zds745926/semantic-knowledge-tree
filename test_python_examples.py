"""
pytest 单元测试 — Python 示例代码

运行:  python -m pytest test_python_examples.py -v
"""
import pytest
import json
import tempfile
import os
from unittest import mock

from python_examples import (
    fibonacci,
    filter_even,
    sort_dict_by_value,
    retry_on_timeout,
    fetch_data,
    fetch_urls_concurrently,
)


# ============================================================
# 测试 1: 斐波那契生成器
# ============================================================

class TestFibonacci:
    def test_first_10(self):
        assert list(fibonacci(10)) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    def test_zero(self):
        assert list(fibonacci(0)) == []

    def test_negative(self):
        assert list(fibonacci(-5)) == []

    def test_one(self):
        assert list(fibonacci(1)) == [0]

    def test_two(self):
        assert list(fibonacci(2)) == [0, 1]


# ============================================================
# 测试 2: 列表推导式过滤偶数
# ============================================================

class TestFilterEven:
    def test_basic(self):
        assert filter_even([1, 2, 3, 4, 5, 6]) == [2, 4, 6]

    def test_all_odd(self):
        assert filter_even([1, 3, 5]) == []

    def test_all_even(self):
        assert filter_even([2, 4, 6]) == [2, 4, 6]

    def test_empty(self):
        assert filter_even([]) == []

    def test_with_negative(self):
        assert filter_even([-2, -1, 0, 1, 2]) == [-2, 0, 2]


# ============================================================
# 测试 3: Lambda 排序字典
# ============================================================

class TestSortDictByValue:
    def test_basic(self):
        result = sort_dict_by_value({'b': 2, 'a': 1, 'c': 3})
        assert result == {'a': 1, 'b': 2, 'c': 3}

    def test_empty(self):
        assert sort_dict_by_value({}) == {}

    def test_ties(self):
        """相同值的键保持相对顺序"""
        result = sort_dict_by_value({'x': 1, 'y': 1, 'z': 2})
        keys = list(result.keys())
        assert keys.index('x') < keys.index('y')  # 插入顺序保留


# ============================================================
# 测试 4: 重试装饰器
# ============================================================

class TestRetryOnTimeout:
    def test_success_first_try(self):
        """装饰器不应干扰正常成功的调用"""
        call_count = 0

        @retry_on_timeout(max_retries=2)
        def ok():
            nonlocal call_count
            call_count += 1
            return 42

        assert ok() == 42
        assert call_count == 1

    def test_retry_then_success(self):
        """失败一次后重试成功"""
        call_count = 0

        @retry_on_timeout(max_retries=3, delay=0.01)
        def eventually_ok():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("模拟超时")
            return "done"

        with mock.patch('time.sleep'):  # 避免实际等待
            assert eventually_ok() == "done"
            assert call_count == 2

    def test_exhaust_retries(self):
        """所有重试耗尽后应抛出异常"""
        call_count = 0

        @retry_on_timeout(max_retries=2, delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("模拟连接失败")

        with mock.patch('time.sleep'):
            with pytest.raises(ConnectionError, match="模拟连接失败"):
                always_fail()
            assert call_count == 3  # 1次原始 + 2次重试

    def test_retry_delay_applied(self):
        """验证每次重试前调用了 time.sleep"""
        @retry_on_timeout(max_retries=2, delay=0.5, backoff=2.0)
        def fail_always():
            raise TimeoutError("超时")

        with mock.patch('time.sleep') as mock_sleep:
            with pytest.raises(TimeoutError):
                fail_always()
            # 第一次 delay=0.5, 第二次 delay=1.0 (backoff 2x)
            assert mock_sleep.call_args_list[0][0][0] == 0.5
            assert mock_sleep.call_args_list[1][0][0] == 1.0


# ============================================================
# 测试 5: fetch_data（mock 网络）
# ============================================================

class TestFetchData:
    @mock.patch('python_examples.requests.get')
    def test_success(self, mock_get):
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"key": "value"}
        mock_get.return_value = mock_response

        result = fetch_data("https://example.com/api")

        assert result == {"key": "value"}
        mock_get.assert_called_once_with("https://example.com/api", timeout=5)

    @mock.patch('python_examples.requests.get')
    def test_retry_on_timeout(self, mock_get):
        """超时后重试"""
        mock_get.side_effect = [
            TimeoutError("timeout"),
            mock.MagicMock(json=lambda: {"ok": True}),
        ]

        with mock.patch('time.sleep'):
            result = fetch_data("https://example.com/api")

        assert result == {"ok": True}
        assert mock_get.call_count == 2


# ============================================================
# 测试 6: 多线程并发请求
# ============================================================

class TestFetchUrlsConcurrently:
    @mock.patch('python_examples.fetch_single')
    def test_all_success(self, mock_fetch):
        mock_fetch.side_effect = [
            ("http://a.com", 200),
            ("http://b.com", 404),
        ]
        results = fetch_urls_concurrently(["http://a.com", "http://b.com"])
        assert len(results) == 2
        assert ("http://a.com", 200) in results
        assert ("http://b.com", 404) in results

    @mock.patch('python_examples.fetch_single')
    def test_partial_failure(self, mock_fetch):
        """某个请求失败时返回 (-1)"""

        def side_effect(url, timeout=10):
            if "bad" in url:
                raise ConnectionError("bad url")
            return (url, 200)

        mock_fetch.side_effect = side_effect

        results = fetch_urls_concurrently(["http://good.com", "http://bad.com"])
        results_dict = dict(results)
        assert results_dict["http://good.com"] == 200
        assert results_dict["http://bad.com"] == -1


# ============================================================
# 测试 7: with 上下文文件读取（集成测试）
# ============================================================

class TestReadFile:
    def test_read(self):
        content = "Hello\nWorld\n"
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        tmp.write(content)
        tmp.close()

        from python_examples import read_file
        result = read_file(tmp.name)
        assert result == content
        os.unlink(tmp.name)
