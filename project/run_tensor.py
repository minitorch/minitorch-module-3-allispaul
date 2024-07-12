"""
Be sure you have minitorch installed in your Virtual Env.
>>> pip install -Ue .
"""

from typing import Any, Callable

import minitorch
from minitorch import Tensor


def RParam(*shape: int) -> minitorch.Parameter:
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers: int):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.layer1.forward(x).relu()
        out2 = self.layer2.forward(out1).relu()
        out3 = self.layer3.forward(out2)
        return out3.sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x: Tensor) -> Tensor:
        batch, dim = x.shape[0], x.shape[1]
        x = x.view(batch, dim, 1)
        out: Tensor = (self.weights.value * x).sum(1) + self.bias.value
        return out.view(batch, self.out_size)


def default_log_fn(epoch: int, total_loss: float, correct: int, losses: Any) -> None:
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers: int):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x: Any) -> Tensor:
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X: Any) -> Tensor:
        return self.model.forward(minitorch.tensor(X))

    def train(
        self,
        data: minitorch.Graph,
        learning_rate: float,
        max_epochs: int = 500,
        log_fn: Callable[[int, float, int, Any], None] = default_log_fn,
    ) -> None:

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE, 200)
