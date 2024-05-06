import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_path = 'model'



# 加载BERT模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
model.eval()

# 翻译结果
translated_sentence = "香港原本有一個人煙稀少嘅漁港, 但係自有英國人嚟咗之後"
# 置信度列表，应与翻译结果的token对应
confidence_scores = [0.819718, 0.81250535, 0.7345596, 0.9116114, 0.53,0.8, 0.8387, 0.7, 0.848,0.93,0.98,0.82,0.81,0.75,0.99,0.89,0.99,0.76,0.49,0.88,0.88,0.88,0.88,0.88,0.88,0.88]

# 将句子转换成token,标点符号不算token
original_tokens = tokenizer.tokenize(translated_sentence)
tokens = original_tokens.copy()





# 遍历token和置信度，对置信度较低的token进行masking
for i, token in enumerate(original_tokens):
    if confidence_scores[i] < 0.7:  # 举例，你可以根据具体阈值调整
        tokens[i] = '[MASK]'
        
print(tokens)
tokens = ['[CLS]'] + tokens + ['[SEP]']
original_tokens = ['[CLS]'] + original_tokens + ['[SEP]']

# 将token转换成input_ids
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)
original_ids = tokenizer.convert_tokens_to_ids(original_tokens)
print(original_ids)
# 将input_ids转换成PyTorch张量
input_ids_tensor = torch.tensor([input_ids])
original_ids_tensor = torch.tensor([original_ids])
# # 使用BERT模型预测缺失的token
with torch.no_grad():
    outputs = model(input_ids_tensor)
    predictions = outputs[0]
print(f"output is {outputs}")
# print(f"predictions is {predictions}")

# 获取预测的token
predicted_tokens = []

# for i, token in enumerate(tokens):
#     print(f"token is {token}")
#     print(f"predictions[0, i] is {predictions[0, i]}")

for i, token in enumerate(tokens):
    if token == '[MASK]':
        # print(predictions[0, i])
        predicted_token_id = torch.argmax(predictions[0, i]).item()
        # print(f"predicted_token_id is {predicted_token_id}")
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
        predicted_tokens.append(predicted_token)
    else:
        predicted_tokens.append(token)

# 将预测的token组合成句子
predicted_sentence = ' '.join(predicted_tokens[1:-1])  # 去除CLS和SEP标记

print("Original Translation:", translated_sentence)
print("Improved Translation:", predicted_sentence)
