import math
import torch
from torch.optim import Optimizer
import SAM

class SSAMD(SAM):
    """
    Sparse SAM с динамическим выпадением и ростом (SSAMD).
    Расширяет SAM добавлением стратегий дропаута и нового роста параметров.
    """

    def __init__(
        self,
        params,
        base_optimizer,
        rho: float,
        sparsity: float,
        drop_rate: float,
        drop_strategy: str,
        growth_strategy: str,
        update_freq: int,
        T_start: int,
        T_end: int,
        **kwargs
    ) -> None:
        """
        params: параметры модели
        base_optimizer: класс или экземпляр Optimizer
        rho: радиус SAM
        sparsity: доля обнуляемых параметров
        drop_rate: максимальная доля дропа за эпоху
        drop_strategy: критерий расчёта "скорости" для дропа ('weight', 'gradient', 'random')
        growth_strategy: критерий для роста ('weight', 'gradient', 'random')
        update_freq: частота (в эпохах) обновления маски
        T_start, T_end: начало и конец цикла изменения drop_rate (эпохи)
        **kwargs: другие аргументы для базового оптимизатора
        """
        # Проверка входных аргументов
        import torch.optim as optim
        assert (
            isinstance(base_optimizer, optim.Optimizer)
            or (isinstance(base_optimizer, type)
                and issubclass(base_optimizer, optim.Optimizer))
        ), "base_optimizer должен быть Optimizer или его класс"

        super().__init__(params, base_optimizer, rho, **kwargs)
        assert rho >= 0.0, f"rho >= 0, получено {rho}"
        assert 0.0 <= sparsity <= 1.0, f"sparsity ∈ [0,1], получено {sparsity}"
        assert 0.0 <= drop_rate <= 1.0, f"drop_rate ∈ [0,1], получено {drop_rate}"
        assert update_freq >= 1, f"update_freq >=1, получено {update_freq}"

        # Параметры SSAMD
        self.sparsity = sparsity
        self.drop_rate = drop_rate
        self.drop_strategy = drop_strategy
        self.growth_strategy = growth_strategy
        self.update_freq = update_freq
        self.T_start = T_start
        self.T_end = T_end

        # Добавляем поля в param_groups
        for group in self.param_groups:
            group.update({
                "sparsity": sparsity,
                "drop_rate": drop_rate,
                "drop_strategy": drop_strategy,
                "growth_strategy": growth_strategy,
                "update_freq": update_freq,
                "T_end": T_end,
                "T_start": T_start
            })

        self.init_mask()

    @torch.no_grad()
    def init_mask(self):
        """
        Инциализируем маску случайным образом, сохраняя top-(1-sparsity) параметров.
        """
        scores = []
        # Генерируем случайные скоры
        for group in self.param_groups:
            for p in group['params']:
                score = torch.rand_like(p)
                self.state[p]['score'] = score.cpu()
                scores.append(score.flatten())

        all_scores = torch.cat(scores)
        keep = int(len(all_scores) * (1 - self.sparsity))
        top_vals, top_idx = torch.topk(all_scores, keep)
        mask_flat = torch.zeros_like(all_scores)
        mask_flat.scatter_(0, top_idx, 1.0)

        # Распределяем обратно по параметрам
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                m = mask_flat[offset:offset+numel].view_as(p)
                self.state[p]['mask'] = m.to(p)
                del self.state[p]['score']
                offset += numel

    @torch.no_grad()
    def DeathRate_Scheduler(self, epoch: int) -> float:
        """
        Косинусный планировщик для динамики drop_rate от T_start до T_end.
        """
        progress = (epoch - self.T_start) / (self.T_end - self.T_start)
        cos_val = (1 + math.cos(math.pi * progress)) / 2
        return self.drop_rate * cos_val

    @torch.no_grad()
    def update_mask(self, epoch: int, **kwargs):
        """
        Обновляем маску: вычисляем death и growth скоры, отбираем параметры методом TopK.
        """
        death_list, growth_list = [], []
        # Считаем скоры
        for group in self.param_groups:
            for p in group['params']:
                death_score = self.get_score(p, self.drop_strategy) * self.state[p]['mask']
                growth_score = self.get_score(p, self.growth_strategy) * (1 - self.state[p]['mask'])
                death_list.append(death_score.flatten())
                growth_list.append(growth_score.flatten())

        death_scores = torch.cat(death_list)
        rate = self.DeathRate_Scheduler(epoch)
        drop_k = int((len(death_scores) - len(death_scores)*self.sparsity) * rate)
        keep_k = int(len(death_scores)*(1-self.sparsity) - drop_k)
        _, death_idx = torch.topk(death_scores, keep_k)

        growth_scores = torch.cat(growth_list)
        _, growth_idx = torch.topk(growth_scores, drop_k)

        total_len = death_scores.numel()
        new_mask = torch.zeros(total_len, device=death_scores.device)
        new_mask.scatter_(0, death_idx, 1.0)
        new_mask.scatter_(0, growth_idx, 1.0)

        # Применяем новую маску к параметрам
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                m = new_mask[offset:offset+numel].view_as(p)
                self.state[p]['mask'] = m.to(p)
                offset += numel

    def get_score(self, p: torch.Tensor, mode: str) -> torch.Tensor:
        device = p.device
        if mode == 'weight':
            return p.abs().detach().to(device)
        elif mode == 'gradient':
            return p.grad.abs().detach().to(device)
        elif mode == 'random':
            return torch.rand_like(p).to(device)
        else:
            raise KeyError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Ascent-шаг SAM с учётом маски: добавляем perturbation только для активных параметров.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-7)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = p.grad * scale
                e_w.mul_(self.state[p]['mask'])
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None, epoch=None, batch_idx=None, logger=None, **kwargs):
        """
        Полный шаг: SAM ascent/descent и обновление маски по расписанию.
        """
        assert closure is not None, "SAM требует closure"
        closure = torch.enable_grad()(closure)

        loss1 = closure()
        self.first_step()

        # Обновляем маску в начале эпохи при batch_idx=0
        if (epoch % self.update_freq == 0) and (batch_idx == 0):
            logger.log("Обновление маски SSAMD")
            self.update_mask(epoch)
            logger.log(f"Доля живых весов: {self.mask_info():.4f}")

        self.zero_grad()
        loss2 = closure()
        self.second_step()
        return loss2

    @torch.no_grad()
    def mask_info(self) -> float:
        """
        Возвращает долю активных (1) элементов маски.
        """
        live = total = 0
        for group in self.param_groups:
            for p in group['params']:
                m = self.state[p]['mask']
                live += m.sum().item()
                total += m.numel()
        return live / total
