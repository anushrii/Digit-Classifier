import torch


class Trainer:
    """
    Initialize training model, hyperparameters, loss & accuracy functions.
    """

    def __init__(self, model, optimizer, loss_fn, accuracy_fn):
        """
        Args:
        model (torch.nn.Module): Neural network model.
        optimizer (torch.optim): Optimizer for the model.
        loss_fn (torch.nn): Loss function.
        accuracy_fn (function): Accuracy function.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn

    def train_step(self, data, labels):
        """
        Perform a single training step.

        Args:
        data (torch.Tensor): Test input data.
        labels (torch.Tensor): Test labels.
        """

        pred = self.model(data)
        loss = self.loss_fn(pred, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        pred_labels = torch.argmax(pred, dim=1)
        accuracy = self.accuracy_fn(labels, pred_labels)
        return loss.item(), accuracy

    def eval_step(self, data, labels):
        """
        Perform a single evaluation step.

        Args:
        data (torch.Tensor): Test input data.
        labels (torch.Tensor): Test labels
        """
        self.model.eval()
        with torch.inference_mode():
            pred = self.model(data)
            loss = self.loss_fn(pred, labels)
            pred_labels = torch.argmax(pred, dim=1)
            accuracy = self.accuracy_fn(labels, pred_labels)

        return loss.item(), accuracy

    def training(self, data_loader):
        """
        Perform training of the model.

        Args:
        data_loader (DataLoader): Training data loader.

        Returns:
        float: Average loss of the model.
        float: Average accuracy of the model.
        """
        self.model.train()
        losses = []
        accuracies = []
        for i, (data, labels) in enumerate(data_loader):

            loss, accuracy = self.train_step(data, labels)
            losses.append(loss)
            accuracies.append(accuracy)

            if i % 100 == 0:
                print(
                    f"Batch: {i} | Train Loss: {loss:.4f} | Train Accuracy: {accuracy:.2f}%"
                )

        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

    def evaluate(self, data_loader):
        """
        Evaluate the model.

        Args:
        data_loader (DataLoader): Evaluation data loader.

        Returns:
        float: Average loss.
        float: Average accuracy.
        """
        self.model.eval()
        losses = []
        accuracies = []
        for i, (data, labels) in enumerate(data_loader):
            loss, accuracy = self.eval_step(data, labels)
            losses.append(loss)
            accuracies.append(accuracy)

            if i % 100 == 0:
                print(
                    f"Batch: {i} | Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.2f}%"
                )

        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)
