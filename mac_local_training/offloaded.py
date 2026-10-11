"""Single backward pass with grouped Adam moments; full gradients stay resident."""
import math
import time
from .synchronous import SynchronousTrainingEngine
from .engine import _json_write, _rss_bytes


class OffloadedTrainingEngine(SynchronousTrainingEngine):
    def _identity(self):
        identity = super()._identity()
        identity['schedule'] = 'single-pass-grouped-adam-v1'
        return identity

    def update_once(self):
        started = time.perf_counter()
        self.model.train()
        self.model.freeze()
        for layer in self.model.layers:
            for name in ('q_proj', 'v_proj'):
                getattr(layer.self_attn, name).unfreeze(keys=['lora_a', 'lora_b'], recurse=False)
        cursor = self.cursor
        accumulated = None
        loss_sum = 0.0
        tokens = 0
        for _ in range(self.config.accumulation_steps):
            rows = [self.train_data[self.order[(cursor + j) % len(self.order)]]
                    for j in range(self.config.microbatch_size)]
            cursor += self.config.microbatch_size
            batch = self._batch(rows)
            loss, grads = self.loss_and_grad(self.model, *batch)
            accumulated = grads if accumulated is None else self.map(lambda a, b: a + b, accumulated, grads)
            self.mx.eval(loss, accumulated)
            loss_sum += float(loss.item())
            tokens += int(batch[2].sum().item())
            del loss, grads, batch
        gradients = self.map(lambda value: value / tokens, accumulated)
        self.mx.eval(gradients)
        flat = dict(self.flatten(gradients))
        del accumulated, gradients
        if not math.isfinite(loss_sum) or any(not bool(self.mx.all(self.mx.isfinite(v)).item()) for v in flat.values()):
            raise FloatingPointError('Non-finite gradients; no update applied')
        gradient_bytes = sum(v.nbytes for v in flat.values())
        peak_resident = 0
        # Force activation even if the last group is already active: the forward
        # above unfroze every adapter, so its trainable tree must be restricted.
        for group_id in range(len(self.groups)):
            if self.active_group == group_id:
                self.model.freeze()
                for layer_id in self.groups[group_id]:
                    for name in ('q_proj', 'v_proj'):
                        getattr(self.model.layers[layer_id].self_attn, name).unfreeze(keys=['lora_a', 'lora_b'], recurse=False)
            else:
                self.activate_group(group_id)
            names = {name for name, _ in self.flatten(self.model.trainable_parameters())}
            selected = self.unflatten([(name, flat[name]) for name in sorted(names)])
            self.optimizer.update(self.model, selected)
            self.mx.eval(self.model.trainable_parameters(), self.optimizer.state)
            peak_resident = max(peak_resident, sum(v.nbytes for _, v in self.flatten(self.optimizer.state)))
            del selected
        del flat
        if any(not bool(self.mx.all(self.mx.isfinite(v)).item()) for v in self.adapter_arrays().values()):
            raise FloatingPointError('Non-finite adapters; checkpoint not committed')
        self.step += 1
        self.cursor = cursor
        self.coverage = [v + 1 for v in self.coverage]
        elapsed = time.perf_counter() - started
        self.training_seconds += elapsed
        self.training_tokens += tokens
        record = {'step': self.step, 'group': 'all_single_pass', 'layers': list(range(len(self.model.layers))),
                  'loss': loss_sum / tokens, 'tokens': tokens, 'seconds': elapsed,
                  'tokens_per_second': tokens / elapsed, 'optimizer_resident_bytes': peak_resident,
                  'trainable_parameters': sum(v.size for v in self.adapter_arrays().values()),
                  'gradient_block_passes': 1, 'full_gradient_resident_bytes': gradient_bytes,
                  'mlx_peak_bytes': int(self.mx.get_peak_memory()), 'process_rss_high_water_bytes': _rss_bytes()}
        self.history.append(record)
        return record

    def run(self):
        report = super().run()
        report['strategy'] = 'single_pass_grouped_adam'
        report['limitations'] = [v for v in report['limitations'] if not v.startswith('Each logical update recomputes')]
        report['limitations'].append('All adapter gradients remain resident during grouped updates; only inactive Adam moments are offloaded.')
        _json_write(self.output / 'report.json', report)
        return report
