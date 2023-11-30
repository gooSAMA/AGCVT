import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn.functional import softmax

# Initial
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


texts = ["I am thrilled to see you again", "This is a terrible experience"]
labels = ["happy", "sad", "angry", "fearful", "surprised", "disgusted"]

# T5_input
t5_input_texts = [f"sentiment: {text}" for text in texts]

# Segment
inputs = tokenizer(t5_input_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

# T5_predict
with torch.no_grad():
    outputs = model.generate(**inputs)

# Output
for i, output in enumerate(outputs):
    decoded_output = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Input text: {texts[i]}")
    print(f"Predicted sentiment: {decoded_output}")
