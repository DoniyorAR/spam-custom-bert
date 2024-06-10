import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from model import ModifiedBERT
from utils import device, Config
from train import SpamDataset
from sklearn.metrics import precision_recall_fscore_support

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

def test():
    df = pd.read_csv(Config.test_data_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_data = SpamDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=Config.max_length
    )

    test_loader = DataLoader(test_data, batch_size=Config.batch_size)

    model = ModifiedBERT().to(device)
    model.load_state_dict(torch.load('spam_model.bin'))

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for d in test_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.tolist())
            all_targets.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
    print(f'Test Precision: {precision*100:.2f}%')
    print(f'Test Recall: {recall*100:.2f}%')
    print(f'Test F1 Score: {f1*100:.2f}%')

if __name__ == '__main__':
    test()
