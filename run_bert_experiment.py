from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer, get_cosine_schedule_with_warmup


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models--bert-base-chinese" / "snapshots" / "8f23c25b06e129b6c986331a13d8d025a92cf0ea"
TRAIN_FILE = BASE_DIR / "train_fixed.tsv"
OUTPUT_DIR = BASE_DIR / "Model_Parameter_save"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_LEN = 30
BATCH_SIZE = 16
EPOCHS = 1
SEED = 2019


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=2):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(self.config.hidden_size, classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        out_pool = outputs.pooler_output
        logit = self.fc(out_pool)
        return logit


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total parameters: {total_num}, Trainable parameters: {trainable_num}"


def load_and_encode(tokenizer):
    input_ids, input_masks, input_types = [], [], []
    labels = []

    with TRAIN_FILE.open(encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f), desc="Encoding train.tsv"):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            y, title = parts
            labels.append(int(y))
            encode_dict = tokenizer(
                title,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
            )
            input_ids.append(encode_dict["input_ids"])
            input_types.append(encode_dict["token_type_ids"])
            input_masks.append(encode_dict["attention_mask"])

            # The teacher's reference code uses a small sample to keep training quick.
            if i > 9000:
                break

    return (
        np.array(input_ids),
        np.array(input_types),
        np.array(input_masks),
        np.array(labels),
    )


def evaluate(model, loader, device, criterion):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for ids, att, tpe, y in loader:
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            logits = model(ids, att, tpe)
            loss = criterion(logits, y)
            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    return float(np.mean(losses)), accuracy_score(labels, preds), labels, preds


def predict_sentiment(model, tokenizer, text, device):
    model.eval()
    encode_dict = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
    )
    infer_input_ids = torch.tensor([encode_dict["input_ids"]], dtype=torch.long).to(device)
    infer_input_masks = torch.tensor([encode_dict["attention_mask"]], dtype=torch.long).to(device)
    infer_input_types = torch.tensor([encode_dict["token_type_ids"]], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(infer_input_ids, infer_input_masks, infer_input_types)
    result = int(np.argmax(output.cpu().numpy(), axis=1)[0])
    return {
        "text": text,
        "label": result,
        "sentiment": "积极" if result == 1 else "消极",
        "logits": output.cpu().numpy()[0].round(4).tolist(),
    }


def main():
    set_seed(SEED)
    print("第一步：确认 bert-base-chinese tokenizer 和本地模型可用")
    print(f"本地模型目录: {MODEL_DIR}")
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    sentence_demo = "今天天气很不错"
    print(tokenizer(sentence_demo))

    print("\n第二步：准备 train.tsv 并编码")
    input_ids, input_types, input_masks, labels = load_and_encode(tokenizer)
    print(input_ids.shape, input_types.shape, input_masks.shape, labels.shape)

    print("\n第三步：打乱数据并划分训练/验证/测试集")
    idxes = np.arange(input_ids.shape[0])
    print("连续数据，打乱前，前10个下标情况：", idxes[:10])
    np.random.seed(SEED)
    np.random.shuffle(idxes)
    print("连续数据，打乱后，前10个下标情况：", idxes[:10])

    input_ids_train, input_ids_valid, input_ids_test = input_ids[idxes[:800]], input_ids[idxes[800:900]], input_ids[idxes[900:]]
    input_masks_train, input_masks_valid, input_masks_test = input_masks[idxes[:800]], input_masks[idxes[800:900]], input_masks[idxes[900:]]
    input_types_train, input_types_valid, input_types_test = input_types[idxes[:800]], input_types[idxes[800:900]], input_types[idxes[900:]]
    y_train, y_valid, y_test = labels[idxes[:800]], labels[idxes[800:900]], labels[idxes[900:]]
    print("训练的input_ids：", input_ids_train.shape, y_train.shape)
    print("验证的input_ids：", input_ids_valid.shape, y_valid.shape)
    print("测试的input_ids：", input_ids_test.shape, y_test.shape)

    print("\n第四步：转换为 DataLoader")
    train_data = TensorDataset(
        torch.LongTensor(input_ids_train),
        torch.LongTensor(input_masks_train),
        torch.LongTensor(input_types_train),
        torch.LongTensor(y_train),
    )
    valid_data = TensorDataset(
        torch.LongTensor(input_ids_valid),
        torch.LongTensor(input_masks_valid),
        torch.LongTensor(input_types_valid),
        torch.LongTensor(y_valid),
    )
    test_data = TensorDataset(
        torch.LongTensor(input_ids_test),
        torch.LongTensor(input_masks_test),
        torch.LongTensor(input_types_test),
    )
    train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)
    valid_loader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=BATCH_SIZE)
    print(f"train_loader={len(train_loader)}, valid_loader={len(valid_loader)}, test_loader={len(test_loader)}")

    print("\n第五步/第六步：定义 BERT 二分类模型并测试一次前向传播")
    model = Bert_Model(str(MODEL_DIR))
    data_iter = iter(train_loader)
    ids, att, tpe, _ = next(data_iter)
    output = model(ids, att, tpe)
    print("data_tensor.shape:", ids.shape)
    print(type(output))
    print(output.shape)
    print(output[0])

    print("\n第七步：打印模型参数")
    print(get_parameter_number(model))
    print("BertModel hidden_size:", model.config.hidden_size)
    print("fc.weight.shape:", tuple(model.fc.weight.shape))

    print("\n第八步：设定训练超参数")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    print("torch version:", torch.__version__)
    print("torch cuda version:", torch.version.cuda)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader),
        num_training_steps=EPOCHS * len(train_loader),
    )
    criterion = nn.CrossEntropyLoss()

    print("\n第九步：开始训练")
    train_history = []
    for i in range(EPOCHS):
        start = time.time()
        model.train()
        print(f"***** Running training epoch {i + 1} *****")
        train_loss_sum = 0.0
        for idx, (ids, att, tpe, y) in enumerate(train_loader):
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            y_pred = model(ids, att, tpe)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item()
            if (idx + 1) % max(1, len(train_loader) // 5) == 0:
                avg_loss = train_loss_sum / (idx + 1)
                print(f"Epoch {i + 1:04d} | Step {idx + 1:04d}/{len(train_loader):04d} | Loss {avg_loss:.4f} | Time {time.time() - start:.4f}")
        train_history.append(train_loss_sum / len(train_loader))

    model_path = OUTPUT_DIR / "best_bert_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

    print("\n验证集评估")
    valid_loss, valid_acc, y_true, y_pred = evaluate(model, valid_loader, device, criterion)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["消极", "积极"], digits=4)
    print(f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")
    print("confusion_matrix:")
    print(cm)
    print("classification_report:")
    print(report)

    print("\n第十步：自定义句子推理")
    samples = [
        "我喜欢你",
        "我不喜欢你",
        "我非常喜欢你",
        "我有点不喜欢你",
        "我对你没什么感觉",
        "这个商品质量很好，下次还会购买",
        "服务太差了，再也不会来了",
    ]
    predictions = [predict_sentiment(model, tokenizer, text, device) for text in samples]
    for item in predictions:
        print(f"{item['text']} -> {item['sentiment']} (label={item['label']}, logits={item['logits']})")

    results = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "data_shape": {
            "input_ids": list(input_ids.shape),
            "input_types": list(input_types.shape),
            "input_masks": list(input_masks.shape),
            "labels": list(labels.shape),
            "train": [len(train_data), int(y_train.shape[0])],
            "valid": [len(valid_data), int(y_valid.shape[0])],
            "test": [len(test_data), int(y_test.shape[0])],
        },
        "train_loss": train_history,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "predictions": predictions,
        "model_path": str(model_path),
    }
    results_path = OUTPUT_DIR / "experiment_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"实验结果已保存到: {results_path}")


if __name__ == "__main__":
    main()
