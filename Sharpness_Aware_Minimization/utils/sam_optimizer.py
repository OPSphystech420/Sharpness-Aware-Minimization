import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def f(x):
    return x**6 + x**5 + 5*x**3 - 30*x**2 + 3*x

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.005, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                if "e_w" not in self.state[p]:
                    self.state[p]["e_w"] = torch.zeros_like(p)
                self.state[p]["e_w"].copy_(e_w)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure):
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        perturbed_loss = closure()
        self.second_step()
        return loss, perturbed_loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        grads = torch.stack([
            p.grad.norm(2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ])
        norm = torch.norm(grads, p=2)
        return norm

x_sam = torch.tensor([0.0], requires_grad=True)
optimizer_sam = SAM([x_sam], torch.optim.SGD, rho=0.005, lr=0.01)


num_epochs = 30
x_range = torch.linspace(-3, 3, 500)
y_range = f(x_range)
frames_sam = []

def sam_closure():
    optimizer_sam.zero_grad()
    loss = f(x_sam)
    loss.backward()
    return loss

for epoch in range(1, num_epochs + 1):
    loss, perturbed_loss = optimizer_sam.step(sam_closure)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_range.detach().numpy(), y_range.detach().numpy(), label=r"$f(x)$", color="blue")
    current_y = f(x_sam).item()
    ax.scatter(x_sam.detach().item(), current_y, color="red", label=f"Current Point: x = {x_sam.detach().item():.3f}, f(x) = {current_y:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"SAM Optimization - Epoch {epoch}")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.legend()
    ax.grid()

    canvas = FigureCanvas(fig)
    canvas.draw()
    frame = np.array(canvas.buffer_rgba())
    frames_sam.append(Image.fromarray(frame))

    plt.close(fig)

frames_sam[0].save("sam_optimization.gif", save_all=True, append_images=frames_sam[1:], duration=200, loop=0)
print("SAM GIF saved as 'sam_optimization.gif'")
