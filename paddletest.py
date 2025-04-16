from paddlenlp import Taskflow

schema = '情感倾向[正向，负向]'
ie = Taskflow('information_extraction', schema=schema, model='uie-base')
ie.set_schema(schema) # Reset schema
print(ie('这个产品用起来真的很流畅，我非常喜欢'))