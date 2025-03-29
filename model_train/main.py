from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
from transformers import pipeline

# 加载数据集
dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)
datasets = dataset.train_test_split(test_size=0.1)
# 加载预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")


# 对模型进行分词处理
def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples


tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
trainset, validset = tokenized_datasets["train"], tokenized_datasets["test"]

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))
for batch in trainloader:
    break
{k: v.shape for k, v in batch.items()}
len(trainloader), len(validloader)
# 加载预训练模型与优化器
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")

if torch.cuda.is_available():
    model = model.cuda()
optimizer = AdamW(model.parameters(), lr=2e-5)

nums_epochs = 3
nums_training_steps = len(trainloader) * nums_epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=nums_training_steps
)
print(nums_training_steps)

# 训练与验证
progress_bar = tqdm(range(nums_training_steps))
# 统一设置device使用的设备 -找到合适的gpu或CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 将模型参数加载的gpu或cpu上
model.to(device)
metric = evaluate.combine(["precision", "f1"])


def evaluate():
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for batch in validloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            pred = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=pred.long(), references=batch["labels"].long())
        # acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return metric.compute()


def train(epoch=2, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:

            # cuda()方法将张量复制到GPU中，从而利用GPU的并行计算能力加速模型训练或推理
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()     # 梯度清零
            output = model(**batch)  # 前向传播
            output.loss.backward()  # 反向传播
            optimizer.step()  # 参数更新

            lr_scheduler.step()  # 更新学习率
            progress_bar.update(1)

            if global_step % log_step == 0:
                print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1
        dev_res = evaluate()
        print(f"ep: {ep}, {dev_res}")


train()
sen = "我觉得这家酒店不错，饭很好吃！"
id2_label = {0: "差评！", 1: "好评！"}
model.eval()
with torch.inference_mode():
    inputs = tokenizer(sen, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1)
    print(f"输入：{sen}\n模型预测结果:{id2_label.get(pred.item())}")
model.config.id2label = id2_label
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
