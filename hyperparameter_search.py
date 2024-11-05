import torch.nn as nn
import torch.optim as optim
import random
from trainer import train, evaluate
from model import SimpleCNN
from data_preparation import get_data_loaders


def grid_search(train_dataset, test_dataset, param_grid, device):
    results = []
    for lr in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            for dropout_rate in param_grid['dropout_rate']:
                print(f'Training with LR: {lr}, Batch Size: {batch_size}, Dropout: {dropout_rate}')

                model = SimpleCNN(dropout_rate=dropout_rate).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()
                train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size)
                
                train(model, train_loader, optimizer, criterion, device, epochs=5)
                accuracy = evaluate(model, test_loader, device)
                
                results.append({
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'dropout_rate': dropout_rate,
                    'accuracy': accuracy
                })
                print(f"Accuracy: {accuracy:.4f}")
    return results


def random_search(train_dataset, test_dataset, param_grid, device, n_trials,):
    results = []
    for _ in range(n_trials):
        lr = random.choice(param_grid['learning_rate'])
        batch_size = random.choice(param_grid['batch_size'])
        dropout_rate = random.choice(param_grid['dropout_rate'])
        
        print(f'Training with LR: {lr}, Batch Size: {batch_size}, Dropout: {dropout_rate}')
        model = SimpleCNN(dropout_rate=dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size)
        
        train(model, train_loader, optimizer, criterion, device, epochs=5)
        accuracy = evaluate(model, test_loader, device)
        
        results.append({
            'learning_rate': lr,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'accuracy': accuracy
        })
        print(f"Accuracy: {accuracy:.4f}")
    return results

