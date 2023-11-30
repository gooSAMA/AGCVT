
from rouge_chinese import Rouge
import jieba # you can use any other word cutting library

hypothesis = "被行政拘留十天"
hypothesis = ' '.join(jieba.cut(hypothesis))

reference = ""

reference = ' '.join(jieba.cut(reference))

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)