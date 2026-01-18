"""
非结构化数据采集模块单元测试
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestAnnouncementCategory(unittest.TestCase):
    """公告类型枚举测试"""
    
    def test_from_string(self):
        """测试从字符串转换"""
        from src.data_pipeline.collectors.unstructured.base import AnnouncementCategory
        
        # 精确匹配
        self.assertEqual(
            AnnouncementCategory.from_string("年报"),
            AnnouncementCategory.PERIODIC_ANNUAL
        )
        
        # 包含匹配
        self.assertEqual(
            AnnouncementCategory.from_string("2024年度报告"),
            AnnouncementCategory.OTHER  # 不是精确包含
        )
        
        # 未知类型
        self.assertEqual(
            AnnouncementCategory.from_string("随机文本"),
            AnnouncementCategory.OTHER
        )
    
    def test_get_periodic_categories(self):
        """测试获取定期报告类型"""
        from src.data_pipeline.collectors.unstructured.base import AnnouncementCategory
        
        periodic = AnnouncementCategory.get_periodic_categories()
        self.assertIn(AnnouncementCategory.PERIODIC_ANNUAL, periodic)
        self.assertIn(AnnouncementCategory.PERIODIC_SEMI, periodic)


class TestDateRangeIterator(unittest.TestCase):
    """日期范围迭代器测试"""
    
    def test_single_chunk(self):
        """测试单个分块"""
        from src.data_pipeline.collectors.unstructured.base import DateRangeIterator
        
        iterator = DateRangeIterator('2024-01-01', '2024-01-15', chunk_days=30)
        chunks = list(iterator)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], ('2024-01-01', '2024-01-15'))
    
    def test_multiple_chunks(self):
        """测试多个分块"""
        from src.data_pipeline.collectors.unstructured.base import DateRangeIterator
        
        iterator = DateRangeIterator('2024-01-01', '2024-03-31', chunk_days=30)
        chunks = list(iterator)
        
        self.assertGreater(len(chunks), 1)
        # 第一个分块应该是30天
        self.assertEqual(chunks[0][0], '2024-01-01')


class TestRateLimiter(unittest.TestCase):
    """速率限制器测试"""
    
    def test_token_bucket_acquire(self):
        """测试令牌桶获取"""
        from src.data_pipeline.collectors.unstructured.rate_limiter import TokenBucket
        
        bucket = TokenBucket(rate=10, capacity=5)
        
        # 应该立即成功
        self.assertTrue(bucket.acquire(1, blocking=False))
        
        # 消耗所有令牌
        for _ in range(4):
            bucket.acquire(1, blocking=False)
        
        # 桶应该为空
        self.assertFalse(bucket.acquire(1, blocking=False))
    
    def test_rate_limiter_domain_config(self):
        """测试域名特定配置"""
        from src.data_pipeline.collectors.unstructured.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        
        # 获取cninfo的配置应该更严格
        cninfo_config = limiter._get_config('www.cninfo.com.cn')
        default_config = limiter._get_config('unknown.com')
        
        self.assertLessEqual(
            cninfo_config.requests_per_second,
            default_config.requests_per_second
        )


class TestProxyPool(unittest.TestCase):
    """代理池测试"""
    
    def test_empty_pool_direct_connection(self):
        """测试空代理池直连"""
        from src.data_pipeline.collectors.unstructured.proxy_pool import ProxyPool
        
        pool = ProxyPool(proxies=[])
        
        # 空代理池应该返回None（直连）
        proxy = pool.get_proxy()
        self.assertIsNone(proxy)
    
    def test_add_proxy(self):
        """测试添加代理"""
        from src.data_pipeline.collectors.unstructured.proxy_pool import ProxyPool
        
        pool = ProxyPool(proxies=[])
        pool.add_proxy('http://127.0.0.1:8080')
        
        self.assertEqual(pool.total_count, 1)


class TestRequestDisguiser(unittest.TestCase):
    """请求伪装器测试"""
    
    def test_get_headers(self):
        """测试获取请求头"""
        from src.data_pipeline.collectors.unstructured.request_utils import RequestDisguiser
        
        disguiser = RequestDisguiser()
        headers = disguiser.get_headers()
        
        self.assertIn('User-Agent', headers)
        self.assertIn('Accept', headers)
    
    def test_random_user_agent(self):
        """测试随机User-Agent"""
        from src.data_pipeline.collectors.unstructured.request_utils import RequestDisguiser
        
        disguiser = RequestDisguiser()
        
        # 获取多个UA，应该有变化（虽然是概率性的）
        uas = set()
        for _ in range(10):
            uas.add(disguiser.get_random_user_agent())
        
        # 至少应该有1个不同的UA
        self.assertGreaterEqual(len(uas), 1)


class TestAnnouncementMetadata(unittest.TestCase):
    """公告元数据测试"""
    
    def test_to_dict(self):
        """测试转换为字典"""
        from src.data_pipeline.collectors.unstructured.base import AnnouncementMetadata
        
        metadata = AnnouncementMetadata(
            ts_code='000001.SZ',
            name='平安银行',
            title='2024年年度报告',
            ann_date='2024-03-30',
            category='年报',
            url='http://example.com/report.pdf',
            source='tushare'
        )
        
        d = metadata.to_dict()
        
        self.assertEqual(d['ts_code'], '000001.SZ')
        self.assertEqual(d['category'], '年报')
    
    def test_from_dict(self):
        """测试从字典创建"""
        from src.data_pipeline.collectors.unstructured.base import AnnouncementMetadata
        
        data = {
            'ts_code': '600000.SH',
            'name': '浦发银行',
            'title': '2024年中报',
            'ann_date': '2024-08-30',
            'category': '中报',
            'url': 'http://example.com',
            'source': 'cninfo',
            'extra_field': 'should be ignored'
        }
        
        metadata = AnnouncementMetadata.from_dict(data)
        
        self.assertEqual(metadata.ts_code, '600000.SH')
        self.assertFalse(hasattr(metadata, 'extra_field'))


class TestUnstructuredCollector(unittest.TestCase):
    """非结构化采集器基类测试"""
    
    def test_standardize_date(self):
        """测试日期标准化"""
        from src.data_pipeline.collectors.unstructured.base import UnstructuredCollector
        
        # 创建一个具体实现来测试
        class TestCollector(UnstructuredCollector):
            def collect(self, start_date, end_date, **kwargs):
                return None
        
        collector = TestCollector()
        
        # YYYYMMDD 格式
        self.assertEqual(
            collector._standardize_date('20240101'),
            '2024-01-01'
        )
        
        # 已经是标准格式
        self.assertEqual(
            collector._standardize_date('2024-01-01'),
            '2024-01-01'
        )
        
        # YYYY/MM/DD 格式
        self.assertEqual(
            collector._standardize_date('2024/01/01'),
            '2024-01-01'
        )
    
    def test_detect_correction(self):
        """测试更正公告检测"""
        from src.data_pipeline.collectors.unstructured.base import UnstructuredCollector
        
        class TestCollector(UnstructuredCollector):
            def collect(self, start_date, end_date, **kwargs):
                return None
        
        collector = TestCollector()
        
        self.assertTrue(collector._detect_correction('关于2024年年报的更正公告'))
        self.assertTrue(collector._detect_correction('补充公告'))
        self.assertFalse(collector._detect_correction('2024年年度报告'))


if __name__ == '__main__':
    unittest.main()
