from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/public/home/yuzhipeng/ChatGLM3-6B", trust_remote_code=True)
from peft import PeftModel
import torch

model = AutoModelForCausalLM.from_pretrained("/public/home/yuzhipeng/ChatGLM3-6B", trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.half, device_map="auto", load_in_8bit=True)
p_model = PeftModel.from_pretrained(model, model_id="./chatglm3_lora/checkpoint-187")

p_model.eval()
print(p_model.chat(tokenizer, "数学考试怎么考高分？", history=[])[0])