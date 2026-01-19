
from src.data_pipeline.collectors.unstructured.policy.miit import MIITCollector
import logging

logging.basicConfig(level=logging.INFO)
collector = MIITCollector()
items = collector._parse_list_page("https://www.miit.gov.cn/search/wjfb.html?websiteid=110000000000000&pg=10&p=1&tpl=14&category=51&q=")
print(f"Captured {len(items)} items")
if items:
    print("First item sample:")
    print(items[0])
collector._close_browser()
