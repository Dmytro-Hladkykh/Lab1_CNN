class Config:
    def __init__(self):
        self.batch_size = 64
        self.lr = 0.001
        self.num_epochs = 10
        self.train_val_split = 0.7  
        self.train_new_model = True
        self.model_path = "trained_model.pth"
