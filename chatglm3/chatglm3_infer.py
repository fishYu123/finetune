

# [gMASK]sop<|user|> \n 考试的技巧有哪些？ <|assistant|> \n Response eos_token
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = AutoTokenizer.from_pretrained("/public/home/yuzhipeng/ChatGLM3-6B", trust_remote_code=True)
ds = Dataset.load_from_disk("./alpaca_data_zh")
print(ds[1])
# def process_func(example):
#     MAX_LENGTH = 256
#     input_ids, attention_mask, labels = [], [], []
#     # query
#     instruction = "\n".join([example['instruction'], example['input']]).strip()
#     # [gMASK]sop<|user|> \n query <|assistant|>
#     instruction = tokenizer.build_chat_input(instruction, history=[], role="user")
#     # \n Response  缺少eos token
#     response = tokenizer("\n" + example['output'], add_special_tokens=False)
#     input_ids = instruction['input_ids'][0].numpy().tolist() + response['input_ids'] + [tokenizer.eos_token_id]
#     attention_mask = instruction['attention_mask'][0].numpy().tolist() + response['attention_mask'] + [1]
#     labels = [-100] * len(instruction['input_ids']) + response['input_ids'] + [tokenizer.eos_token_id]
#     if len(input_ids) > MAX_LENGTH:
#         input_ids = input_ids[:MAX_LENGTH]
#         attention_mask = attention_mask[:MAX_LENGTH]
#         labels = labels[:MAX_LENGTH]
#
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }

def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = "\n".join([example["instruction"], example["input"]]).strip()     # query
    instruction = tokenizer.build_chat_input(instruction, history=[], role="user")  # [gMASK]sop<|user|> \n query<|assistant|>
    response = tokenizer("\n" + example["output"], add_special_tokens=False)        # \n response, 缺少eos token
    input_ids = instruction["input_ids"][0].numpy().tolist() + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"][0].numpy().tolist() + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"][0].numpy().tolist()) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
print(tokenized_ds)
print(tokenizer.decode(tokenized_ds[1]['input_ids']))
print(tokenizer.decode(list(filter(lambda x:x!=-100, tokenized_ds[1]["labels"]))))

## 创建模型
model = AutoModelForCausalLM.from_pretrained("/public/home/yuzhipeng/ChatGLM3-6B", trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.half, device_map="auto", load_in_8bit=True)
from peft import LoraConfig, TaskType, get_peft_model
config = LoraConfig(target_modules=["query_key_value"])
model = get_peft_model(model, config)

## 配置参数
args = TrainingArguments(output_dir="./chatglm3_lora",
                         per_device_train_batch_size=4,
                         gradient_accumulation_steps=8,
                         logging_steps=10,
                         num_train_epochs=1,
                         learning_rate=1e-4,
                         remove_unused_columns=False,
                         save_strategy="epoch")

trainer = Trainer(model=model,
                args=args,
                train_dataset=tokenized_ds.select(range(6000)),
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))
#
trainer.train()
#

#


#
#
#
#
#
#
#
#
#





















