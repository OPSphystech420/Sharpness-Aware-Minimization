import torch
from torch.optim import Optimizer

class FriendlySAM(Optimizer):
    """
    Sharpness-Aware Minimization с дружелюбным направлением градиента.
    Добавляет EMA и корректирует направление шага.
    """

    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        sigma: float = 1.0,
        lmbda: float = 0.9,
        adaptive: bool = False,
        **kwargs
    ):
        """
        params: параметры модели
        base_optimizer: класс оптимизатора (torch.optim.SGD и т.п.)
        rho: радиус perturbation для SAM
        sigma: коэффициент весового вычитания EMA из градиента
        lmbda: коэффициент экспоненциального сглаживания для EMA
        adaptive: флаг адаптивного SAM
        **kwargs: lr, momentum, weight_decay и др. для base_optimizer
        """
        assert rho >= 0.0, f"rho должно быть >= 0, получено: {rho}"
        # Инициализируем Optimizer
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        # Базовый оптимизатор и его параметры
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        # Параметры FriendlySAM
        self.sigma = sigma
        self.lmbda = lmbda

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        Первый шаг SAM: корректируем градиент через EMA, затем ascent-перемещение.
        """
        # Проходим по всем группам параметров
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                state = self.state[p]
                # Инициализация EMA m_t
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                state['momentum'].mul_(self.lmbda).add_(g, alpha=1 - self.lmbda)
                # Вычисляем дружелюбное направление d_t = g - sigma * m_t
                d = g - self.sigma * state['momentum']
                p.grad = d

        # Стандартный ascent SAM
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                # Сохраняем старые веса
                self.state[p]['old_p'] = p.data.clone()
                coeff = torch.abs(p) if group['adaptive'] else 1.0
                e_w = coeff * p.grad * scale.to(p.dtype)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Второй шаг SAM: возвращаем веса и вызываем базовый optimizer.step().
        """
        # Восстанавливаем исходные веса перед descent
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']
        # Шаг базового оптимизатора
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """
        Полный шаг SAM: ascent и descent через две фазы closure().
        closure: функция, выполняющая forward+backward.
        """
        assert closure is not None, "Для SAM требуется closure"
        closure = torch.enable_grad()(closure)
        # Первая фаза: вычисляем градиенты
        loss = closure()
        # Ascent с дружелюбным направлением
        self.first_step(zero_grad=True)
        # Вторая фаза: переоценка градиентов
        loss = closure()
        # Descent
        self.second_step()
        return loss

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """
        Считает L2-норму градиента по всем параметрам,
        с учётом adaptive SAM, если включён.
        """
        device = self.param_groups[0]['params'][0].device
        norms = []

        for group in self.param_groups:
            adaptive = group.get('adaptive', False)
            for p in group['params']:
                if p.grad is None:
                    continue
                # compute coefficient per-parameter
                coeff = torch.abs(p) if adaptive else 1.0
                norms.append((coeff * p.grad).norm(p=2).to(device))

        # stack all and take overall L2-norm
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        """
        Загружает состояние optimizer и синхронизирует param_groups базового optimizer.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
