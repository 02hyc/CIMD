import paddle
import json
from paddlenlp import Taskflow

class SentimentAnalyzer:
    def __init__(self, device='gpu'):
        paddle.set_device(device)
        self.schema = '情感倾向[正向，负向]'
        self.ie = Taskflow('information_extraction', schema=self.schema, model='uie-base', device=device)
        self.ie.set_schema(self.schema)
    
    def get_sentiment_score(self, text):
        """获取文本的情感分数，范围(-1,1)"""
        result = self.ie(text)
        if not result or not result[0].get('情感倾向[正向，负向]'):
            return 0.0
        
        sentiment = result[0]['情感倾向[正向，负向]'][0]
        score = sentiment['probability']
        # 如果是负向情感，将分数转为负数
        if sentiment['text'] == '负向':
            score = -score
        return score

# 加载成语释义数据
def load_idiom_definitions(file_path='/home/yukino/Downloads/NLP/CIMD/DEBERT/Idiom/Idiom_Definition.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 初始化情感分析器
analyzer = SentimentAnalyzer(device='gpu')

# 测试函数
def analyze_text_with_idiom(text, idiom, idiom_def):
    # 分析原句情感
    text_sentiment = analyzer.get_sentiment_score(text)
    # 分析成语释义情感
    def_sentiment = analyzer.get_sentiment_score(idiom_def)
    
    return {
        'text': text,
        'text_sentiment': text_sentiment,
        'idiom': idiom,
        'idiom_definition': idiom_def,
        'definition_sentiment': def_sentiment
    }

# 示例使用
if __name__ == '__main__':
    # 加载成语释义
    # idiom_defs = load_idiom_definitions()
    
    # 测试样例
    test_text = '他在工作中兢兢业业，深受领导好评'
    result = analyzer.get_sentiment_score(test_text)
    print(f'原句子: {test_text}')
    print(f'情感分数: {result}')
    
    # 成语分析示例
    # if idiom_defs:
    #     example_idiom = '兢兢业业'
    #     if example_idiom in idiom_defs:
    #         example_def = idiom_defs[example_idiom]['definition']
    #         result = analyze_text_with_idiom(test_text, example_idiom, example_def)
    #         print('\n成语分析结果:')
    #         for k, v in result.items():
    #             print(f'{k}: {v}')