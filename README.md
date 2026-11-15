# Backpropagation from Scratch

A small multilayer perceptron trained with an explicit gradient step, so each part of backpropagation stays visible: the forward pass, the loss, the chain rule, and the weight update.

The scalar autograd engine is Andrej Karpathy’s [micrograd](https://github.com/karpathy/micrograd), from his lecture [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0). That engine is vendored as-is. The training loop in `main.py` is mine: the data, the loss, the gradient reset, and the parameter update.

## Model

`MLP(3, [4, 4, 1])`:

- 3 inputs
- two hidden layers of 4 neurons, ReLU
- 1 linear output

Every weight and bias is a micrograd `Value`: a scalar, a gradient, and the operation that produced it. The network has 41 parameters.

Training uses four examples, the toy set from the lecture:

| Input | Target |
|---|---|
| `[2.0, 3.0, -1.0]` | `1.0` |
| `[3.0, -1.0, 0.5]` | `-1.0` |
| `[0.5, 1.0, 1.0]` | `-1.0` |
| `[1.0, 1.0, -1.0]` | `1.0` |

## Training step

Each of 100 steps does four things:

1. **Forward.** Evaluate the network on all four inputs.
2. **Loss.** Sum the squared error between each prediction and its target.
3. **Backward.** `loss.backward()` walks the computation graph in reverse and applies the chain rule. Each parameter receives `grad`: how the loss changes if that parameter moves.
4. **Update.** Gradients are set to `0` before the backward pass so they do not accumulate. Each parameter then steps against its gradient: `data += -0.015 * grad`.

That update is plain gradient descent at learning rate `0.015`. No optimizer library is involved.

## Result

Weights start from `Uniform(-1, 1)`, so the opening loss changes between runs. On the last run it began near `3.5` and finished near `0.008`. The four predictions landed close to `1, -1, -1, 1`.

## Run

Python 3.12. No third-party packages. micrograd is the local copy under `micrograd/`, and the script uses that plus the standard library.

```bash
python main.py
```

The script prints the loss at each step, then the four final predictions.
