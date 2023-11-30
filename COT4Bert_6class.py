import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Initialize
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)


texts = ["I am thrilled to see you again", "This is a terrible experience"]
labels = ["happy", "sad", "angry", "fearful", "surprised", "disgusted"]  # 示例标签

# COT
cot_texts = [f"COT: Analyzing sentiment of '{text}'. Sentiment is " for text in texts]

# Segment
inputs = tokenizer(cot_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    predictions = softmax(outputs.logits, dim=1)

# Output
for cot_text, prediction in zip(cot_texts, predictions):
    sentiment = labels[torch.argmax(prediction)]
    print(f"{cot_text} {sentiment}")
