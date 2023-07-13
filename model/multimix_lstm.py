import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        targets,
        num_categories,
        task_weights=None,
        mixed_task_name: str = None,
        censored_task_label: int = -1,
        dropout_rate=0.0,
        pos_weight=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_categories = num_categories
        self.num_tasks = len(output_size)
        self.output_size = output_size
        self.task_weights = task_weights
        self.targets = targets
        self.mixed_task_name = mixed_task_name
        self.mixed_task_idx = self.targets.index(self.mixed_task_name)
        self.censored_task_label = censored_task_label
        self.dropout_rate = dropout_rate
        self.pos_weight = pos_weight

        self.embedding = nn.Embedding(num_categories, hidden_size)
        self.lstm = nn.LSTM(
            input_size + hidden_size, hidden_size, num_layers, batch_first=True
        )
        if self.num_tasks == 1:
            self.fc = nn.Linear(hidden_size, output_size[0])
        else:
            self.fc_hidden = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size) for i in range(self.num_tasks)]
            )
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.fc_out = nn.ModuleList(
                [nn.Linear(hidden_size, output_size[i]) for i in range(self.num_tasks)]
            )
        if self.num_tasks > 2:
            raise NotImplementedError("Only 2 tasks supported at this moment")

    def forward(self, x):
        embedded = self.embedding(x["encoder_cat"]).squeeze()
        x = x["encoder_cont"]
        x = torch.cat((x, embedded), dim=-1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        if self.num_tasks == 1:
            out = self.fc(out[:, -1, :])
        else:
            out = torch.stack(
                [
                    self.fc_out[i](self.dropout(self.fc_hidden[i](out[:, -1, :])))
                    for i in range(self.num_tasks)
                ],
                dim=-1,
            )
        return out

    def training_step_dev(self, batch, batch_idx):
        """
        Development training step (i.e. is not actually being used for training but good as placeholder for some other code)
        """
        x, y = batch
        y = y[0]
        if self.num_tasks == 1:
            y_hat = self(x)
            loss = nn.BCEWithLogitsLoss()(y_hat, y[0])
        else:
            loss = 0
            y_hat = self(x)
            loss += nn.BCEWithLogitsLoss()(y_hat[:, 0], y[0])
            loss += nn.MSELoss()(y_hat[:, 1], y[1])
        self.log("train_loss", loss)
        return loss

    def validation_step_dev(self, batch, batch_idx):
        """
        Development validation step (i.e. is not actually being used for validation but good as placeholder for some other code)
        """
        x, y = batch
        y = y[0]
        if self.num_tasks == 1:
            y_hat = self(x)
            loss = nn.BCEWithLogitsLoss()(y_hat, y[0])
        else:
            loss = 0
            y_hat = self(x)
            loss += nn.CrossEntropyLoss()(y_hat[:, 0], y[0])
            loss += nn.MSELoss()(y_hat[:, 1], y[1])

        self.log("val_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x["encoder_cont"].shape[0]
        y = y[0]
        if self.num_tasks == 1:
            y_hat = self(x)
            loss = {}
            loss["mixed"] = nn.BCEWithLogitsLoss(
                reduction="mean", pos_weight=torch.tensor([self.pos_weight])
            )(
                y_hat,
                y.float(),
            )
            final_loss = loss["mixed"]
        else:
            y_hat = self(x)
            task_mask = y[self.mixed_task_idx] != self.censored_task_label
            # Check if any instances have valid targets for task 1
            loss = {}
            if torch.any(task_mask):
                # Compute loss for task 1 with the mask
                loss["mixed"] = nn.BCEWithLogitsLoss(
                    reduction="mean", pos_weight=torch.tensor([self.pos_weight])
                )(
                    y_hat[:, :, self.mixed_task_idx][task_mask],
                    y[self.mixed_task_idx].float()[task_mask],
                )
                # Enable parameter update for task 1
                self.fc_hidden[self.mixed_task_idx].weight.requires_grad = True
                self.fc_hidden[self.mixed_task_idx].bias.requires_grad = True
                self.fc_out[self.mixed_task_idx].weight.requires_grad = True
                self.fc_out[self.mixed_task_idx].bias.requires_grad = True
            else:
                # Set loss_task1 to 0 if no valid targets
                loss["mixed"] = torch.tensor(0.0).to(y[self.mixed_task_idx].device)

                # Switch off parameter update for task 1
                self.fc_hidden[self.mixed_task_idx].weight.requires_grad = False
                self.fc_hidden[self.mixed_task_idx].bias.requires_grad = False
                self.fc_out[self.mixed_task_idx].weight.requires_grad = False
                self.fc_out[self.mixed_task_idx].bias.requires_grad = False

            loss["regular"] = [
                nn.MSELoss()(y_hat[:, :, i], y[i].float())
                for i in range(self.num_tasks)
                if i != self.mixed_task_idx
            ][0]
            final_loss = [loss["mixed"], loss["regular"]]
            final_loss = torch.sum(
                torch.stack(final_loss, dim=0)
                * torch.tensor(self.task_weights).to(y[0].device)
            )
            self.log(
                "train_loss_reg",
                loss["regular"],
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            self.log(
                "train_loss",
                final_loss,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        self.log(
            "train_loss_mixed",
            loss["mixed"],
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return final_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x["encoder_cont"].shape[0]
        y = y[0]
        if self.num_tasks == 1:
            y_hat = self(x)
            loss = {}
            # Compute loss for task 1 with the mask
            loss["mixed"] = nn.BCEWithLogitsLoss(reduction="mean")(
                y_hat,
                y.float(),
            )
            self.log_metrics(
                y_hat,
                y.float(),
                batch_size,
            )
            final_loss = loss["mixed"]
        else:
            y_hat = self(x)
            task_mask = y[self.mixed_task_idx] != self.censored_task_label
            # Check if any instances have valid targets for task 1
            loss = {}
            if torch.any(task_mask):
                # Compute loss for task 1 with the mask
                loss["mixed"] = nn.BCEWithLogitsLoss(reduction="mean")(
                    y_hat[:, :, self.mixed_task_idx][task_mask],
                    y[self.mixed_task_idx].float()[task_mask],
                )
                self.log_metrics(
                    y_hat[:, :, self.mixed_task_idx][task_mask],
                    y[self.mixed_task_idx].float()[task_mask],
                    batch_size,
                )
            else:
                # Set loss_task1 to 0 if no valid targets
                loss["mixed"] = torch.tensor(0.0).to(y[self.mixed_task_idx].device)

            loss["regular"] = [
                nn.MSELoss()(y_hat[:, :, i], y[i].float())
                for i in range(self.num_tasks)
                if i != self.mixed_task_idx
            ][0]
            final_loss = [loss["mixed"], loss["regular"]]
            final_loss = torch.sum(
                torch.stack(final_loss, dim=0)
                * torch.tensor(self.task_weights).to(y[0].device)
            )
            self.log(
                "val_loss_reg",
                loss["regular"],
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                batch_size=batch_size,
            )
            self.log(
                "val_loss",
                final_loss,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                batch_size=batch_size,
            )
        self.log(
            "val_loss_mixed",
            loss["mixed"],
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=batch_size,
        )

        return final_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def log_metrics(self, y_hat, y, batch_size):
        # Compute probability and binary prediction
        y_prob = torch.sigmoid(y_hat)
        y_pred = (y_prob > 0.5).float()

        # Compute metrics
        accuracy = torchmetrics.Accuracy(task="binary")(y_pred, y)
        precision = torchmetrics.Precision(average="weighted", task="binary")(y_pred, y)
        recall = torchmetrics.Recall(average="weighted", task="binary")(y_pred, y)
        f1 = torchmetrics.F1Score(average="weighted", task="binary")(y_pred, y)

        # Log metrics
        self.log(
            "val_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_precision",
            precision,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_recall",
            recall,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        self.log(
            "val_f1",
            f1,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=batch_size,
        )
