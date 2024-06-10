class Config:
    random_seed = 42
    train_data_path = 'train.csv'
    test_data_path = 'test.csv'
    model_path = 'spam_model.bin'
    max_length = 128
    batch_size = 32
    learning_rate = 2e-5
    epochs = 3

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
