from sklearn.metrics import accuracy_score
from torch import sigmoid
import torch


class Trainer:

    def __init__(self, model, criterion, device, optimizer, batch_size, n_epochs, learning_rate) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

    def load_data(self, train_dataloader, test_dataloader) -> None:
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def get_accuracy(preds, labels) -> float:
        preds = (sigmoid(preds) > 0.5).float() 

        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        return accuracy
    
    def _train_step(self, texts, labels) -> tuple:
        texts, labels = texts.to(self), labels.to(self)

        self.optimizer.zero_grad()

        preds = self.model(texts)
        loss = self.criterion(preds, labels.float())
        loss.backward()
        acc = self.get_accuracy(preds, labels)

        self.optimizer.step()
        return float(loss.cpu().detach().numpy()), float(acc)
    
    def _train_epoch(self, data):
        epoch_loss = []
        epoch_acc = []
        
        for texts, labels in data:
            loss, acc = self.train_step(self.model, texts, labels)
            epoch_loss.append(loss)
            epoch_acc.append(acc)
            
        return epoch_loss, epoch_acc
    
    def _test_epoch(self, model, data):
        self.model.eval()
        
        epoch_loss = []
        epoch_acc = []
        
        with torch.no_grad():
            for texts, labels in data:
                loss, acc = self.train_step(texts, labels)
                epoch_loss.append(loss)
                epoch_acc.append(acc)
            
        return epoch_loss, epoch_acc
    
    def loop(model, optimizer, criterion, train_dataloader, test_dataloader, n_epochs):
        train_epoch_losses = []
        test_epoch_losses = []
        train_epoch_acces = []
        test_epoch_acces = []
        
        for epoch in range(n_epochs):
            for loader, is_train in zip([train_dataloader, test_dataloader], [True, False]):
                batch_iterator = tqdm(loader, unit="batch", leave=True)
                
                epoch_loss, epoch_acc = train_epoch(model, optimizer, criterion, batch_iterator)

                train_str = f"Epoch {epoch+1}/{n_epochs} " + f"{'Train' if is_train else 'Test'} Loss: {np.mean(epoch_loss)/(batch_iterator.n+1):.4f} " +f"{'Train' if is_train else 'Test'} Accuracy: {np.mean(epoch_acc)/(batch_iterator.n+1):.4f}" 

                if is_train:
                    train_epoch_losses.append(epoch_loss)
                    train_epoch_acces.append(epoch_acc)
                else: 
                    test_epoch_losses.append(epoch_loss)
                    test_epoch_acces.append(epoch_acc)

    #                 batch_iterator.set_description(train_str)
                batch_iterator.set_postfix_str({
                    'epoch' : f'{epoch+1}/{n_epochs}',
                    'stage' : 'train' if is_train else 'test',
                    'loss'  : f'{np.mean(epoch_loss):.4f}',
                    'acc'   : f'{np.mean(epoch_acc):.4f}'
                })
                batch_iterator.update()
    #                 break
    #         break
        history = [
            train_epoch_losses, test_epoch_losses, train_epoch_acces, test_epoch_acces
        ]
        return history   