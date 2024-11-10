import unittest
from src.quality.content_checker import ContentChecker

class TestContentChecker(unittest.TestCase):
    def setUp(self):
        self.checker = ContentChecker()
        self.sample_news = """【测试新闻标题】
2024年3月20日，某地发生重大事件。
据相关部门通报，此次事件造成严重影响。
目前，相关部门正在积极处理此事。
"""
        self.sample_info = {
            'time': '2024年3月20日',
            'location': '某地',
            'event_type': '重大事件',
            'impact': '造成严重影响',
            'measures': '积极处理'
        }

    def test_basic_check(self):
        passed, issues = self.checker.check_content(self.sample_news, self.sample_info)
        self.assertTrue(passed)
        self.assertEqual(len(issues), 0)

    def test_forbidden_words(self):
        news_with_forbidden = self.sample_news.replace('发生', '据说发生')
        passed, issues = self.checker.check_content(news_with_forbidden, self.sample_info)
        self.assertFalse(passed)
        self.assertTrue(any('据说' in issue for issue in issues))

    def test_missing_fields(self):
        info_missing_field = self.sample_info.copy()
        del info_missing_field['impact']
        passed, issues = self.checker.check_content(self.sample_news, info_missing_field)
        self.assertFalse(passed)
        self.assertTrue(any('impact' in issue for issue in issues))

    def test_content_density(self):
        low_density_news = """【测试新闻】
的的的的的的的的的的的的的的的的的的的的。
了了了了了了了了了了了了了了了了了了了了。
"""
        passed, issues = self.checker.check_content(low_density_news, self.sample_info)
        self.assertFalse(passed)
        self.assertTrue(any('密度' in issue for issue in issues)) 