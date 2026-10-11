"""Post-hoc allocation trace; not a confirmation run or a speed benchmark.

Instrument trusted, hash-pinned project code with memory observations after
accumulation, normalization and optimizer application. Core math is unchanged.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalization_study import PACKAGE, digest, verify, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--method', choices=['endpoint', 'normalized', 'rootreleased'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.study / 'protocol.json').read_text())
    verify(protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    name = {'normalized': 'normalization.py', 'endpoint': 'offloaded.py', 'rootreleased': 'rootrelease.py'}[args.method]
    source = (PACKAGE / 'mac_local_training' / name).read_text()
    substitutions = [
        ('            self.mx.eval(loss, accumulated)\n', "            self.mx.eval(loss, accumulated)\n            self.observe_phase('accumulation', accumulated{roots})\n"),
        ('            self.mx.eval(self.model.trainable_parameters(), self.optimizer.state)\n', "            self.mx.eval(self.model.trainable_parameters(), self.optimizer.state)\n            self.observe_phase('optimizer_application')\n")]
    substitutions = [(a, b.replace('{roots}', '' if args.method == 'rootreleased' else ', grads')) for a, b in substitutions]
    if args.method != 'normalized':
        substitutions.append(('        self.mx.eval(gradients)\n', "        self.mx.eval(gradients)\n        self.observe_phase('full_normalization', accumulated, gradients)\n"))
        class_name = 'RootReleaseTrainingEngine' if args.method == 'rootreleased' else 'OffloadedTrainingEngine'
    else:
        substitutions.append(('            self.mx.eval([value for _, value in selected_flat])\n', "            self.mx.eval([value for _, value in selected_flat])\n            self.observe_phase('group_normalization', flat, selected_flat)\n"))
        class_name = 'StreamingNormalizationEngine'
    for old, new in substitutions:
        if source.count(old) != 1:
            raise ValueError('Trace point no longer uniquely matches frozen source')
        source = source.replace(old, new)
    instrumented = args.output / 'instrumented.py'
    instrumented.write_text(source)
    spec = importlib.util.spec_from_file_location('mac_local_training._allocation_profile', instrumented)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from mac_local_training.engine import TrainingConfig
    from mac_local_training.endpoint import endpoint_order
    import mlx.core as mx
    parent = getattr(module, class_name)
    class Observed(parent):
        def observe_phase(self, phase, *trees):
            if self.step < 2:
                return
            self.events.append({'phase': phase, 'active_bytes': int(mx.get_active_memory()),
                                'cumulative_peak_bytes': int(mx.get_peak_memory()),
                                'observed_gradient_tree_buffer_bytes': int(mx.get_array_buffer_size(*trees)) if trees else None})
        def update_group_order(self):
            return endpoint_order(len(self.groups), self.step)
    cfg = TrainingConfig(train_path=protocol['train']['path'], valid_path=protocol['valid']['path'],
                         output_path=str(args.output / 'training'), model_path=protocol['model_path'],
                         strategy='rotating', steps=3, rank=256, seed=17, **protocol['config'])
    engine = Observed(cfg)
    engine.events = []
    for _ in range(2):
        engine.update_once()
    mx.eval(engine.model.parameters(), engine.optimizer.state)
    mx.reset_peak_memory()
    engine.observe_phase('start')
    engine.update_once()
    engine.observe_phase('end')
    result = {'status': 'post-hoc diagnostic; not used for confirmation gates', 'method': args.method,
              'seed': 17, 'rank': 256, 'profiled_update': 3, 'events': engine.events,
              'original_source_sha256': digest(PACKAGE / 'mac_local_training' / name),
              'instrumented_source_sha256': digest(instrumented),
              'source_protocol_sha256': digest(args.study / 'protocol.json'),
              'limitations': ['Two warmups and one instrumented update on one seed; no speed comparison.',
                              'Phase observations are cumulative high-water readings at these boundaries, not a complete allocator timeline.',
                              'Array buffer sizes exclude graph objects, caches and non-buffer memory.']}
    write(args.output / 'trace.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
