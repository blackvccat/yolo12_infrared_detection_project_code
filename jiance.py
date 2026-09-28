import torch

model_path = r"C:\Users\blackvccat\PycharmProjects\PythonProject\ultralytics\runs\detect\neko_model6\weights\last.pt"
ckpt = torch.load(model_path, map_location="cpu")
print("模型加载成功")
