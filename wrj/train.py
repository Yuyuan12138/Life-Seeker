import torch
import numpy as np
import random
import pandas as pd

from auto_loss_fn import CustomLoss
import data
from model import Net
from config import Config
config = Config()

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# set_seed(3407)
# print(config.data_name)
# exit(0)

print(f'Trying to load data from {config.data_name}')
train_loader, test_loader = data.get_dataloader(train_path=config.data_path + config.data_name + '/train.tsv',
                                                test_path=config.data_path + config.data_name + '/test.tsv')
print('Loaded data')

model = Net(4, 2)
model.to(config.device)
print('Created model on', config.device)

criterion = torch.nn.CrossEntropyLoss()
criterion = CustomLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

train_loss_table = []
train_acc_table = []
test_loss_table = []
test_acc_table = []

print('Start training')
max_acc = 0
for epoch in range(config.epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    total_step = len(train_loader)
    for i, train_data in enumerate(train_loader):
        inputs, labels = train_data

        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels, 0.5)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

        train_accuracy = train_correct / train_total

        if(i == total_step-1):
            train_loss /= len(train_data)
            train_loss_table.append(train_loss)
            train_acc_table.append(train_accuracy)
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}({train_correct}/{train_total})')

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            inputs, labels = test_data

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels, 0.5)

            val_loss += loss.item()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    max_acc = max(val_accuracy, max_acc)
    avg_val_loss = val_loss / len(test_data)

    test_loss_table.append(avg_val_loss)
    test_acc_table.append(val_accuracy)
    print(f'Epoch [{epoch+1}/{config.epochs}], Val Loss: {avg_val_loss:.4f}, Max Accuracy: {max_acc:.4f}({val_correct}/{val_total})\n')

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion_state_dict': criterion.state_dict()
}, 'model.pth')

# loss/acc可视化
import matplotlib.pyplot as plt

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(list(range(1, config.epochs+1)), train_loss_table, label='Train', color='blue')
plt.plot(list(range(1, config.epochs+1)), test_loss_table, label='Test', color='orange')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(list(range(1, config.epochs+1)), train_acc_table, label='Train', color='blue')
plt.plot(list(range(1, config.epochs+1)), test_acc_table, label='Test', color='orange')
plt.ylim(0, 1)
plt.legend()
plt.title('Accuracy')

plt.suptitle(config.data_name)
plt.show()
plt.savefig(f'./figures/{config.data_name}-{config.loop}.jpg')

if config.loop != 114514:
    with open('results.csv', 'a+') as f:
        f.write(f'{config.data_name},{config.loop},{max_acc}\n')