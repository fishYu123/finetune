from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from tqdm import  tqdm
from peft import LoraConfig, TaskType, get_peft_model
import torch

tokenizer = AutoTokenizer.from_pretrained("/public/home/yuzhipeng/ChatGLM3-6B", trust_remote_code=True)

# 数据读取
data = pd.read_json("../data/train/lora/caocao.json")
datasets = Dataset.from_pandas(data)

# 数据处理
def process_func(example):
    print(example)
    MAX_LENGTH = 128
    input_ids, attention_mask, labels = [], [], []
    instruction = "\n".join([example['instruction'], example['input']]).strip()
    instruction = tokenizer.build_chat_input(instruction, history=[], role="user")
    response = tokenizer("\n" + example['input'], add_special_tokens=False)
    input_ids = instruction['input_ids'][0].numpy().tolist() + response['input_ids'] + [tokenizer.eos_token_id]
    attention_mask = instruction['attention_mask'][0].numpy().tolist() + response['attention_mask'] + [1]
    labels = [-100] * len(instruction['input_ids'][0].numpy().tolist()) + response['input_ids'] + [tokenizer.eos_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]


    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized_ds = datasets.map(process_func, remove_columns=datasets.column_names)

print(tokenizer.decode(tokenized_ds[0]["input_ids"]))
print(tokenizer.decode(tokenized_ds[0]['labels']))
## 创建模型
model = AutoModelForCausalLM.from_pretrained("/public/home/yuzhipeng/ChatGLM3-6B",trust_remote_code=True, torch_dtype=torch.half, low_cpu_mem_usage=True,  load_in_8bit=True)
config = LoraConfig(target_modules=["query_key_value"])
model = get_peft_model(model, config)

## 配置参数
train_args = TrainingArguments(
    output_dir="./CaocaoModel",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    logging_steps=10,
    num_train_epochs=5,
    learning_rate=5e-5,
    remove_unused_columns=False)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()