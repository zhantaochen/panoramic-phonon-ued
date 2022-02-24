import abc
import math
import torch
from .event_handling import find_event
from .misc import _handle_unused_kwargs

DEBUG_MODE = False
torch.set_printoptions(precision=10)

class AdaptiveStepsizeODESolver(metaclass=abc.ABCMeta):
    def __init__(self, dtype, y0, norm, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.y0 = y0
        self.dtype = dtype

        self.norm = norm

    def _before_integrate(self, t):
        pass

    @abc.abstractmethod
    def _advance(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        t = t.to(self.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i] = self._advance(t[i])
        return solution


class AdaptiveStepsizeEventODESolver(AdaptiveStepsizeODESolver, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _advance_until_event(self, event_fn):
        raise NotImplementedError

    def integrate_until_event(self, t0, event_fn):
        t0 = t0.to(self.y0.device, self.dtype)
        self._before_integrate(t0.reshape(-1))
        event_time, y1 = self._advance_until_event(event_fn)
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution


class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, save_all_timepoints=False, requires_grad=False, true_states=None, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb
        # added the argument "save_all_timepoints" for returning all time points values
        self.save_all_timepoints = save_all_timepoints
        self.requires_grad = requires_grad
        self.true_states = true_states

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            # niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            # t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            # t_infer[-1] = t[-1]

            niters = math.ceil(abs(end_time.item() - start_time.item()) / step_size)
            t_infer = torch.linspace(start_time, end_time, niters+1, dtype=t.dtype, device=t.device)

            return t_infer
        return _grid_constructor

    def _grid_list_constructor_from_step_size(self, t):
        time_grid_list = []
        time_list = [t[i:i+2] for i in range(len(t) - 1)]
        for _t in time_list:
            time_grid = self.grid_constructor(self.func, self.y0, _t)
            assert time_grid[0] == _t[0] and time_grid[-1] == _t[-1]
            assert torch.all((time_grid[1:] - time_grid[:-1]) <= (self.step_size + 1e-4)), \
                print(time_grid[1:] - time_grid[:-1], self.step_size + 1e-4)
            time_grid_list.append(time_grid)
        return time_grid_list

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t, shapes=None):
        # time_grid = self.grid_constructor(self.func, self.y0, t)
        # assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        if self.true_states is not None:
            # time_grid_list = [-_time_grid.flip(dims=(0,)) for _time_grid in reversed(self.true_states["time_grid_list"])]
            assert torch.allclose(self.true_states["time_grid"][[0,-1]], self.grid_constructor(self.func, self.y0, t)[[0,-1]]), \
                print(self.true_states["time_grid"], '\n', self.grid_constructor(self.func, self.y0, t))
            time_grid_list = [self.true_states["time_grid"], ]
        else:
            time_grid_list = self._grid_list_constructor_from_step_size(t)
        
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        if self.save_all_timepoints:
            solution_all_timepoints = [torch.empty(len(time_grid), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device) for time_grid in time_grid_list]

        j = 1
        y0 = self.y0
        for i_time_grid, time_grid in enumerate(time_grid_list):
            if DEBUG_MODE:
                print(time_grid)
            if self.save_all_timepoints:
                solution_all_timepoints[i_time_grid][0] = y0.detach()
                k = 1
            for i_t, (t0, t1) in enumerate(zip(time_grid[:-1], time_grid[1:])):
                dt = t1 - t0 

                # if (shapes is not None) and (self.true_states is not None):
                #     if i_t == 0:
                #         assert torch.allclose(
                #             y0[shapes[0].numel() : (shapes[0].numel() + shapes[1].numel())], 
                #             self.true_states["y_grid"][i_t].detach().reshape(-1)
                #         )
                #     if i_t < len(time_grid)-1:
                #         y0[shapes[0].numel() : (shapes[0].numel() + shapes[1].numel())] = \
                #             self.true_states["y_grid"][i_t].detach().reshape(-1).clone()

                with torch.no_grad():
                    dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
                y1 = y0 + dy
                
                if self.true_states is None and DEBUG_MODE:
                    print(f"time: {t0}")
                    print(y0[0,:,0,0,-1])
                    print(dy[0,:,0,0,-1])
                
                if (shapes is not None) and (self.true_states is not None):
                    if i_t == 0:
                        assert torch.allclose(
                            y0[shapes[0].numel() : (shapes[0].numel() + shapes[1].numel())], 
                            self.true_states["y_grid"][i_t].detach().reshape(-1)
                        )
                    
                    if i_t < len(time_grid)-1:
                        y1[shapes[0].numel() : (shapes[0].numel() + shapes[1].numel())] = \
                            self.true_states["y_grid"][i_t+1].detach().reshape(-1).clone()

                if self.save_all_timepoints:
                    solution_all_timepoints[i_time_grid][k] = y1.detach()
                    k += 1

                while j < len(t) and t1 >= t[j]:
                    if self.interp == "linear":
                        solution[j] = self._linear_interp(t0, t1, y0, y1, t[j]).detach()
                    elif self.interp == "cubic":
                        f1 = self.func(t1, y1)
                        solution[j] = self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j]).detach()
                    else:
                        raise ValueError(f"Unknown interpolation method {self.interp}")
                    j += 1
                y0 = y1

        if isinstance(solution, list):
            solution = torch.cat(solution, dim=0)
        if self.save_all_timepoints:
            # solution_all_timepoints = torch.cat(solution_all_timepoints, dim=0)
            return (solution, solution_all_timepoints, time_grid_list)
        else:
            return solution

    def integrate_until_event(self, t0, event_fn):
        assert self.step_size is not None, "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0)
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
