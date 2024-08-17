"""
Sample code for distillation using Hugging Face Transformers. This code was generated from ChatGPT and corrected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# 1. Load the Teacher and Student Models
teacher_model_name = "bert-base-uncased"  # Replace with an actual available model
student_model_name = "distilbert-base-uncased"  # A smaller model for distillation

teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name, num_labels=2)
student_model = AutoModelForSequenceClassification.from_pretrained(student_model_name, num_labels=2)

teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# 2. Load and Tokenize the Dataset
dataset = load_dataset("imdb", split="train[:1000]")  # Small sample for demonstration

def tokenize_data(examples):
    return student_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_data, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 3. Define the Distillation Loss Function
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    ce_loss = F.cross_entropy(student_logits, labels)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return alpha * ce_loss + (1 - alpha) * kl_loss

# 4. Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)

train_loader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True)
optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-5)

num_epochs = 3
teacher_model.eval()  # Freeze teacher model during distillation
student_model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Get student predictions
        student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Calculate distillation loss
        loss = distillation_loss(student_logits, teacher_logits, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed with average loss: {avg_loss:.4f}")

# 5. Evaluate the Student Model
test_dataset = load_dataset("imdb", split="test[:1000]")  # Small sample for demonstration
tokenized_test_dataset = test_dataset.map(tokenize_data, batched=True)
tokenized_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_loader = DataLoader(tokenized_test_dataset, batch_size=16, shuffle=False)

student_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Student model accuracy: {accuracy:.2f}")