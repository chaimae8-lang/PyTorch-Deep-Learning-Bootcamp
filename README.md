# PyTorch-Deep-Learning-Bootcamp

A clean, production-grade repository containing core supervised learning workflows built from scratch using **PyTorch**. This project implements and compares fundamental neural network paradigms covering **Linear Regression** and non-linear **Data Classification** (binary circles and multi-class blobs) without external high-level abstraction wrappers.

---

## 🔬 Core Workflows Covered

### 1. Linear Regression Workflow
- Built using an object-oriented class pattern subclassing `nn.Module` with raw `nn.Parameter` tracking.
- Successfully models continuous linear equations using an `nn.L1Loss` cost function and stochastic gradient descent optimization (`torch.optim.SGD`).
- Verifies exact weight and bias parameter convergence through step-by-step backpropagation routines.

### 2. Binary & Multi-class Data Classification
- Designed neural network graphs utilizing `nn.Sequential` wrappers containing chained `nn.Linear` and non-linear `nn.ReLU` activations.
- Handles complex geometric decision boundaries (synthetic circles and four-cluster blobs) with exact structural shape-matching.
- Optimized multi-class loss computations using `nn.CrossEntropyLoss` with strict target casting to `torch.LongTensor`.

---

## 🛠️ Key Architectural & Memory Optimizations

- **Autograd Cycle Preservation:** Avoided standard runtime errors by systematically enforcing `optimizer.zero_grad()` at the head of every epoch, blocking artificial gradient accumulation.
- **VRAM Leak Prevention:** Extracted real-time performance evaluation metrics strictly via `.item()` to unlink scalar losses from the computational graph, saving extensive GPU hardware overhead.
- **High-Performance Evaluation:** Deployed `torch.inference_mode()` context managers during the test loops to turn off gradient tracking, resulting in significant execution throughput gains.
- **Logit Reduction Pipeline:** Streamlined multi-class evaluation by extracting prediction indices directly via `torch.argmax` on raw continuous logit structures, avoiding unnecessary Softmax exponents in memory.
