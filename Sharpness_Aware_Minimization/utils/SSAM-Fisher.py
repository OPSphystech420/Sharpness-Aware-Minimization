import torch
from torch.optim import Optimizer
import SAM

class SSAMF(SAM):
    """
    Sparse Sharpness-Aware Minimization с фильтрацией по Фишеру.
    Наследует SAM, добавляя регулярное обновление разрежающей маски.
    """

    def __init__(
        self,
        params,
        base_optimizer,
        rho: float,
        sparsity: float,
        num_samples: int,
        update_freq: int,
        criterion,
        **kwargs
    ):
        """
        params: параметры модели
        base_optimizer: класс базового оптимизатора (torch.optim.SGD и т.п.)
        rho: радиус SAM
        sparsity: доля «обнуляемых» направлений (0–1)
        num_samples: число точек для оценки Фишера
        update_freq: частота (в эпохах) обновления маски
        criterion: функция потерь для оценки Фишера
        **kwargs: lr, momentum, weight_decay и остальные для base_optimizer
        """
        super().__init__(params, base_optimizer, rho, **kwargs)

        assert 0.0 <= sparsity <= 1.0, f"sparsity ∈ [0,1], получено: {sparsity}"
        assert num_samples >= 1, f"num_samples ≥ 1, получено: {num_samples}"
        assert update_freq >= 1, f"update_freq ≥ 1, получено: {update_freq}"

        self.sparsity = sparsity
        self.num_samples = num_samples
        self.update_freq = update_freq
        self.criterion = criterion

        # Обогащаем группы параметров метаданными
        for g in self.param_groups:
            g.update(dict(
                sparsity=sparsity,
                num_samples=num_samples,
                update_freq=update_freq
            ))

        self.init_mask()

    @torch.no_grad()
    def init_mask(self):
        """
        Инициализирует нулевую маску для каждого параметра.
        """
        for g in self.param_groups:
            for p in g['params']:
                self.state[p]['mask'] = torch.zeros_like(p)

    @torch.no_grad()
    def update_mask(self, model, train_data):
        """
        Оценивает значимость параметров по Фишеру и обновляет маску.
        """
        fisher = {id(p): torch.zeros_like(p) for g in self.param_groups for p in g['params']}
        was_train = model.training
        model.eval()

        loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
        with torch.enable_grad():
            for i, (x, y) in enumerate(loader):
                if i >= self.num_samples:
                    break
                x, y = x.cuda(), y.cuda()
                out = model(x)
                loss = self.criterion(out, y).mean()
                loss.backward()
                for g in self.param_groups:
                    for p in g['params']:
                        fisher[id(p)] += p.grad.square()
                model.zero_grad()
        if was_train:
            model.train()

        all_vals = torch.cat([v.flatten() for v in fisher.values()])
        k = int(len(all_vals) * (1 - self.sparsity))
        _, idx = torch.topk(all_vals, k)

        mask_flat = torch.zeros_like(all_vals)
        mask_flat[idx] = 1.0

        offset = 0
        for g in self.param_groups:
            for p in g['params']:
                n = p.numel()
                self.state[p]['mask'] = mask_flat[offset:offset+n].view_as(p).to(p.device)
                offset += n

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Ascent-шаг SAM с применением текущей маски.
        """
        grad_norm = self._grad_norm()
        for g in self.param_groups:
            scale = g['rho'] / (grad_norm + 1e-12)
            for p in g['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = p.grad * scale
                e_w.mul_(self.state[p]['mask'])
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None, model=None, train_data=None, epoch=None, batch_idx=None, logger=None):
        """
        Выполняет SAM-шаг и обновляет маску при необходимости.
        """
        assert closure is not None, "Для SSAMF нужен closure"
        assert model is not None and train_data is not None
        assert epoch is not None and batch_idx is not None and logger is not None

        # Ascend + Descent через SAM
        loss = super().step(closure)

        # Обновление маски в начале эпохи
        if (epoch % self.update_freq == 0) and (batch_idx == 0):
            logger.log("Обновление маски SSAMF")
            self.update_mask(model, train_data)
            logger.log(f"Доля живых весов: {self.mask_info():.4f}")

        return loss

    @torch.no_grad()
    def mask_info(self) -> float:
        """
        Доля активных элементов маски.
        """
        live = total = 0
        for g in self.param_groups:
            for p in g['params']:
                m = self.state[p]['mask']
                live += m.sum().item()
                total += m.numel()
        return live / total
