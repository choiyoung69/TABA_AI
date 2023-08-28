from transformers import BertModel
from keybert import KeyBERT
import torch.nn as nn
import torch

class CustomBertModel(nn.Module):
    def __init__(self, keybert_model):
        super(CustomBertModel, self).__init__()
        self.keybert_model = keybert_model

    def forward(self, inputs):
        input_text = inputs[0]  # inputs의 구조에 따라 수정
        self.input = self.keybert_model.extract_keywords(input_text)
        self.keywords, self.weights = zip(*self.input)
        # 키워드를 패딩하여 모두 동일한 길이로 만듦
        max_keyword_len = max(len(keyword) for keyword in self.keywords)

        # 이 부분을 수정하여 키워드와 가중치를 직접 PyTorch 텐서로 생성
        keyword_tensors = [keyword.ljust(max_keyword_len) for keyword in self.keywords]
        keyword_tensors = torch.Tensor([keyword.encode('utf-8') for keyword in keyword_tensors])
        weight_tensors = torch.Tensor(self.weights).unsqueeze(1)
        
        return [torch.cat((keyword_tensors, weight_tensors), dim=1)]

model = BertModel.from_pretrained("skt/kobert-base-v1")
model
keybert_model = KeyBERT(model)

input_text = ["Your input text goes here."]
inputs = keybert_model.extract_keywords(input_text)
keywords, weights = zip(*inputs)

# 이 부분은 이전과 동일하게 생성
keyword_tensors = torch.Tensor([keyword.encode('utf-8') for keyword in keywords])
weight_tensors = torch.Tensor(weights).unsqueeze(1)

# CustomBertModel에 입력으로 사용할 텐서 생성
combined_tensor = torch.cat((keyword_tensors, weight_tensors), dim=1)

custom_model = CustomBertModel(keybert_model)
traced_model = torch.jit.trace(custom_model, [combined_tensor, ])
