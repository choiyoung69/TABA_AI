import torch
from transformers import BertModel
from keybert import KeyBERT

# KeyBERT 모델 준비
model = BertModel.from_pretrained("skt/kobert-base-v1")
keybert_model = KeyBERT(model)


keybert_model = keybert_model.eval()
#script 방식
model_input = torch.rand(1, 64, 64)
traced = torch.jit.trace(keybert_model, (model_input,))

# TorchScript 모델 저장
traced.save("model.pt")
