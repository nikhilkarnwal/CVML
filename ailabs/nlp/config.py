class Config:
    device = 'cuda'
    train = True
    eval = True
    data_path = 'abc.txt'
    vocab_path = 'vocab.json'
    lr = 0.001
    val_path = 'val.json'
    dir = 'saved'

    # config
    epochs = 1
    data_worker = 6
    batch_size = 1
    hidden_dim = 1024
    num_layers = 1
    drop_prob = 0.0

    # other
    resume = 'model.pth'
