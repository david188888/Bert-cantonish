import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW

class ASRDataset(Dataset):
    def __init__(self, tokenizer, texts, masks, labels):
        self.tokenizer = tokenizer
        self.texts = texts
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        mask = self.masks[idx]
        label = self.labels[idx]

        inputs = self.tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # Apply confidence mask: Only allow model to attend to high-confidence tokens
        attention_mask = attention_mask * torch.tensor(mask)

        # Encode labels
        label_ids = self.tokenizer(label, max_length=512, padding='max_length', truncation=True, return_tensors='pt')['input_ids'].squeeze()

        return input_ids, attention_mask, label_ids

# Initialize tokenizer and model
model_path = 'model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)

# Example data
texts = ["我今天去公图书馆"]
confidences = [[0.9, 0.8, 0.6, 0.4, 0.8]]
labels = ["我今天去图书馆"]
masks = [[1 if score > 0.5 else 0 for score in confidence] for confidence in confidences]

# Create dataset and dataloader
dataset = ASRDataset(tokenizer, texts, masks, labels)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(3):  # Train for 3 epochs
    for input_ids, attention_mask, label_ids in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_ids = label_ids.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=label_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
