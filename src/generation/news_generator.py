from jinja2 import Template
from datetime import datetime

class NewsGenerator:
    def __init__(self):
        self.templates = {
            'natural_disaster': {
                'earthquake': Template("""
【地震快讯】{{ time }}，{{ location }}发生{{ event_type }}。
初步测定：震级{{ magnitude }}级，震源深度{{ depth }}千米。
{{ impact }}
目前，当地政府已启动应急预案，{{ measures }}。
请当地居民保持警惕，关注官方信息。
                """.strip()),
                
                'flood': Template("""
【洪水预警】{{ time }}，{{ location }}遭遇{{ event_type }}。
{{ impact }}
防汛部门提醒：{{ measures }}
请市民提高警惕，做好防范。
                """.strip()),
                
                'default': Template("""
【自然灾害】{{ time }}，{{ location }}发生{{ event_type }}。
{{ impact }}
目前，当地政府已启动应急预案，{{ measures }}。
记者将持续关注事态发展。
                """.strip())
            },
            'accident': Template("""
【突发事故快讯】{{ time }}，{{ location }}发生{{ event_type }}事故。
{{ impact }}
目前，相关部门正在{{ measures }}。
后续进展本台将继续报道。
            """.strip()),
            
            'public_health': Template("""
【公共卫生事件】{{ time }}，{{ location }}发生{{ event_type }}。
初步调查显示，{{ impact }}
相关部门已采取措施：{{ measures }}
请公众保持关注，做好防护。
            """.strip()),
            
            'social_security': Template("""
【社会安全事件】{{ time }}，{{ location }}发生{{ event_type }}事件。
据了解，{{ impact }}
当地警方表示，{{ measures }}
案件正在进一步处理中。
            """.strip()),
            
            'other': Template("""
【最新消息】{{ time }}，{{ location }}发生{{ event_type }}。
{{ impact }}
相关部门表示，{{ measures }}
            """.strip())
        }
    
    def generate_news(self, event_type, event_info):
        """生成新闻报道"""
        event_info = self._validate_event_info(event_info)
        
        # 处理具体灾害类型
        if event_type == 'natural_disaster':
            if '地震' in event_info['event_type']:
                template = self.templates['natural_disaster']['earthquake']
            elif '洪水' in event_info['event_type']:
                template = self.templates['natural_disaster']['flood']
            else:
                template = self.templates['natural_disaster']['default']
        else:
            template = self.templates.get(event_type, self.templates['other'])
        
        return template.render(**event_info)
    
    def _validate_event_info(self, event_info):
        """验证和补充事件信息"""
        default_info = {
            'time': datetime.now().strftime('%Y年%m月%d日%H时%M分'),
            'location': '相关地区',
            'event_type': '突发事件',
            'impact': '具体影响正在评估中',
            'measures': '正在采取相应措施'
        }
        
        # 使用默认值补充缺失信息
        for key, default_value in default_info.items():
            if key not in event_info or not event_info[key]:
                event_info[key] = default_value
        
        return event_info