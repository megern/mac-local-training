"""Experimental synchronous block Adam: recompute gradients, defer all updates.

Unlike rotating training, every block uses the same examples and pre-update
weights in each logical update. Gradients and inactive moments are stored on
disk. This trades extra backward passes and I/O for reduced resident optimizer
state; it does not make a larger frozen base model fit. No novelty claim.
"""
import math
from pathlib import Path
import tempfile
import time
from .engine import TrainingEngine, _json_write, _rss_bytes


class SynchronousTrainingEngine(TrainingEngine):
    def __init__(self, config):
        if config.strategy != 'rotating' or config.window_updates != 1:
            raise ValueError('Synchronous engine requires rotating window layout and window_updates=1')
        super().__init__(config)

    def _identity(self):
        identity = super()._identity()
        identity['schedule'] = 'synchronous-deferred-block-adam-v1'
        return identity

    def update_once(self):
        started = time.perf_counter()
        self.model.train()
        batches = []
        cursor = self.cursor
        for _ in range(self.config.accumulation_steps):
            batches.append([self.train_data[self.order[(cursor + j) % len(self.order)]]
                            for j in range(self.config.microbatch_size)])
            cursor += self.config.microbatch_size
        rng = [self.mx.array(value) for value in self.mx.random.state]
        self.mx.eval(rng)
        loss_sum = None
        token_count = None
        peak_resident = 0
        gradient_bytes = 0
        with tempfile.TemporaryDirectory(prefix='pending-gradients-', dir=self.output) as temp:
            directory = Path(temp)
            # No adapter update is permitted until every block's gradient is saved.
            for group_id in range(len(self.groups)):
                self.activate_group(group_id)
                self.mx.random.state = [self.mx.array(value) for value in rng]
                accumulated = None
                block_loss = 0.0
                block_tokens = 0
                for rows in batches:
                    batch = self._batch(rows)
                    loss, grads = self.loss_and_grad(self.model, *batch)
                    accumulated = grads if accumulated is None else self.map(lambda a, b: a + b, accumulated, grads)
                    self.mx.eval(loss, accumulated)
                    block_loss += float(loss.item())
                    block_tokens += int(batch[2].sum().item())
                    del loss, grads, batch
                gradients = self.map(lambda value: value / block_tokens, accumulated)
                self.mx.eval(gradients)
                if not math.isfinite(block_loss) or any(not bool(self.mx.all(self.mx.isfinite(value)).item())
                        for _, value in self.flatten(gradients)):
                    raise FloatingPointError('Non-finite block gradients; no update applied')
                if loss_sum is None:
                    loss_sum, token_count = block_loss, block_tokens
                elif token_count != block_tokens or not math.isclose(loss_sum, block_loss, rel_tol=1e-5, abs_tol=1e-5):
                    raise ValueError('Block forwards disagree; synchronous update refused')
                flat = dict(self.flatten(gradients))
                gradient_bytes += sum(value.nbytes for value in flat.values())
                self.mx.save_safetensors(str(directory / f'{group_id}.safetensors'), flat)
                peak_resident = max(peak_resident, sum(value.nbytes for _, value in self.flatten(self.optimizer.state)))
                del flat, gradients, accumulated
            # A failure in this phase leaves no new checkpoint. Resume must use
            # the previous committed checkpoint, not the partially updated object.
            for group_id in range(len(self.groups)):
                self.activate_group(group_id)
                flat = self.mx.load(str(directory / f'{group_id}.safetensors'))
                gradients = self.unflatten(list(flat.items()))
                self.optimizer.update(self.model, gradients)
                self.mx.eval(self.model.trainable_parameters(), self.optimizer.state)
                peak_resident = max(peak_resident, sum(value.nbytes for _, value in self.flatten(self.optimizer.state)))
                del flat, gradients
        if any(not bool(self.mx.all(self.mx.isfinite(value)).item()) for value in self.adapter_arrays().values()):
            raise FloatingPointError('Non-finite adapters; checkpoint not committed')
        self.step += 1
        self.cursor = cursor
        self.coverage = [value + 1 for value in self.coverage]
        elapsed = time.perf_counter() - started
        self.training_seconds += elapsed
        self.training_tokens += token_count
        record = {'step': self.step, 'group': 'all_synchronous', 'layers': list(range(len(self.model.layers))),
                  'loss': loss_sum / token_count, 'tokens': token_count, 'seconds': elapsed,
                  'tokens_per_second': token_count / elapsed, 'optimizer_resident_bytes': peak_resident,
                  'trainable_parameters': sum(value.size for value in self.adapter_arrays().values()),
                  'gradient_block_passes': len(self.groups), 'temporary_gradient_bytes': gradient_bytes,
                  'mlx_peak_bytes': int(self.mx.get_peak_memory()), 'process_rss_high_water_bytes': _rss_bytes()}
        self.history.append(record)
        return record

    def run(self):
        report = super().run()
        report['strategy'] = 'synchronous_deferred_block_adam'
        report['limitations'] = [value for value in report['limitations']
                                if not value.startswith(('Rotating windows', 'Training throughput'))]
        report['limitations'].extend([
            'Each logical update recomputes the same batch for every block; extra compute and disk I/O are required.',
            'Training timing includes block state/gradient I/O; it excludes checkpoint I/O and validation.',
            'This implements synchronous optimizer offload/recomputation; it makes no algorithm priority claim.',
            'After a failed partial apply, discard the engine and resume the last committed checkpoint.'])
        _json_write(self.output / 'report.json', report)
        return report
