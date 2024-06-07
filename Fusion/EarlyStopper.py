class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_acc = 0.85

    def early_stop(self, val_acc):
        if val_acc > (self.min_val_acc + self.min_delta):
            self.min_val_acc = val_acc
            self.counter = 0
        elif val_acc < (self.min_val_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        elif val_acc >= 0.99:
            return True
        return False
