from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained('/public/home/yuzhipeng/ChatGLM3-6B', trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained('/public/home/yuzhipeng/ChatGLM3-6B', trust_remote_code=True, device='cuda')
model = model.cuda()
model = torch.nn.DataParallel(model, device_ids=None)
print(model.device_ids)


p_model = PeftModel.from_pretrained(model.module, model_id = "./fine-tune/AllModel/checkpoint-195")
p_model.eval()


# his = []
# for i in range(10):
#     question = "朕要更衣了"
#     res, his = p_model.chat(tokenizer, question, history=his)
#     print("甄嬛：" + res + "\n")
his= []
while True:
    question = input("User:")
    res, his = p_model.chat(tokenizer, question, history=his)
    print("Actor：" + res + "\n")
