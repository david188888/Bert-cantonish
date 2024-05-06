import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_path = 'model'

model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# 翻译结果
right_sentense = ' 香港原本係一個人煙稀少嘅漁港, 但係自從英國人嚟咗之後'
wrong_sentense = ' 香港原本有一個人煙稀少嘅漁港, 但係自有英國人嚟咗之後'

# 将句子转化为token
right_tokens = tokenizer.tokenize(right_sentense)
wrong_tokens = tokenizer.tokenize(wrong_sentense)


wrong_sentense_confidence_scores = confidence_scores = [0.819718, 0.81250535, 0.7345596, 0.9116114, 0.53,0.8, 0.8387, 0.7, 0.848,0.93,0.98,0.82,0.81,0.75,0.99,0.89,0.99,0.76,0.49,0.88,0.88,0.88,0.88,0.88,0.88,0.88]

# 遍历token和置信度，对置信度较低的token进行masking
for i, token in enumerate(wrong_tokens):
    if wrong_sentense_confidence_scores[i] < 0.7:
        wrong_tokens[i] = '[MASK]'
        
        
# 将token转换成input_ids
right_input_ids = tokenizer.convert_tokens_to_ids(right_tokens)
wrong_input_ids = tokenizer.convert_tokens_to_ids(wrong_tokens)

# 将input_ids转换成PyTorch张量
right_input_ids_tensor = torch.tensor([right_input_ids])
wrong_input_ids_tensor = torch.tensor([wrong_input_ids])

# 计算output 里面的loss
with torch.no_grad():
    output = model(right_input_ids_tensor, labels=wrong_input_ids_tensor)
    loss = output.loss
    print(f"loss is {loss}")
    print(f"output is {output}")
    
    
    
        