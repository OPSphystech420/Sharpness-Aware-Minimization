import torch
from torch.optim import Optimizer

class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        params: параметры модели
        base_optimizer: класс базового оптимизатора (например, torch.optim.SGD)
        rho: радиус шага в пространстве весов
        adaptive: флаг адаптивного масштабирования (ASAM)
        **kwargs: дополнительные аргументы для базового оптимизатора
        """
        assert rho >= 0.0, f"Неверное значение rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        # Инициализируем базовый оптимизатор
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # Синхронизируем группы параметров и их дефолты
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Первый «восходящий» шаг: поднимаемся по градиенту на расстояние rho.
        Если adaptive=True, коэффициент пропорционален абсолютному значению весов.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Сохраняем старые веса для восстановления
                self.state[p]["old_p"] = p.data.clone()
                coef = torch.abs(p) if group["adaptive"] else 1.0
                e_w = coef * p.grad * scale.to(p.dtype)
                p.add_(e_w)  # w ← w + e(w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Второй «нисходящий» шаг: возвращаем оригинальные веса и делаем шаг
        базовым оптимизатором с учётом новых градиентов.
        """
        # Восстанавливаем исходные веса
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        # Шаг базового оптимизатора
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """
        Полный шаг SAM: вычисляем градиенты, делаем ascent, пересчёт градиентов и descent.
        closure: функция, выполняющая полный проход (forward + backward).
        """
        assert closure is not None, "Для SAM требуется передать closure"
        # Включаем градиенты внутри closure
        closure = torch.enable_grad()(closure)

        # 1) Вычисляем начальную потерю и градиенты
        loss = closure()
        # 2) Атакующий шаг (ascent)
        self.first_step(zero_grad=True)
        # 3) Пересчитываем градиенты в точке w + e(w)
        loss = closure()
        # 4) Нисходящий шаг (descent)
        self.second_step()
        return loss

    def _grad_norm(self):
        """
        Подсчёт нормы градиента для всех параметров.
        При adaptive=True используем взвешенный градиент.
        """
        device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            coef = None
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Выбор коэффициента для adaptive SAM
                coef = torch.abs(p) if group["adaptive"] else 1.0
                norms.append((coef * p.grad).norm(p=2).to(device))
        # Общая L2-норма
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        """
        Загружает состояние оптимизатора и синхронизирует param_groups у базового оптимизатора.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
