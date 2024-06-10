import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import pandas as pd
from model import ModifiedBERT
from utils import device, Config
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import Adam

class CustomAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CustomAttention, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.context_vector = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, hidden_states):
        attention_scores = self.tanh(self.dense(hidden_states))
        attention_scores = torch.matmul(attention_scores, self.context_vector).squeeze(-1)
        attention_weights = self.softmax(attention_scores)
        weighted_output = torch.einsum('bld,bl->bd', hidden_states, attention_weights)
        return weighted_output, attention_weights

class ModifiedBERT(nn.Module):
    def __init__(self):
        super(ModifiedBERT, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased', num_hidden_layers=6)
        self.bert = BertModel(config)
        self.custom_attention = CustomAttention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        attention_output, _ = self.custom_attention(last_hidden_state)
        logits = self.classifier(attention_output)
        return logits

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        return {
          'text': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0
    total = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        total += labels.size(0)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.double() / total, sum(losses) / len(losses)

def train():
    df = pd.read_csv(Config.train_data_path)
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=Config.random_seed)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_data = SpamDataset(
        texts=df_train.text.to_numpy(),
        labels=df_train.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=Config.max_length
    )
    val_data = SpamDataset(
        texts=df_val.text.to_numpy(),
        labels=df_val.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=Config.max_length
    )

    train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.batch_size)

    model = ModifiedBERT().to(device)
    optimizer = Adam(model.parameters(), lr=Config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(Config.epochs):
        print(f'Epoch {epoch + 1}/{Config.epochs}')
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)

        print(f'Train loss {train_loss} accuracy {train_acc}')

    torch.save(model.state_dict(), 'spam_model.bin')

if __name__ == '__main__':
    train()
