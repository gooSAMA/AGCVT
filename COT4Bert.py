import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Initialize
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


texts = ["I love this product", "This is a terrible experience"]

# COT_hardprompt
cot_texts = [f"COT: Analyzing sentiment of '{text}'. Sentiment is " for text in texts]

# Segment COT_augment_content
inputs = tokenizer(cot_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

# Prediction_by BERT
with torch.no_grad():
    outputs = model(**inputs)
    predictions = softmax(outputs.logits, dim=1)

# Output
for cot_text, prediction in zip(cot_texts, predictions):
    sentiment = 'positive' if torch.argmax(prediction) == 1 else 'negative'
    print(f"{cot_text} {sentiment}")
