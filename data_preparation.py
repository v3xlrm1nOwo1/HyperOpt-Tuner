import torch.utils.data as data



def get_data_loaders(train_dataset, test_dataset, batch_size):
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
