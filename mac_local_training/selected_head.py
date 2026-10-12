"""Experimental supervised-position output projection for completion losses.

The transformer sees the entire prompt. Only masked-in positions enter the
vocabulary projection. This is an established sparse-supervision optimization,
not a new optimizer. Kernel-shape changes require a numerical audit.
"""
from .rootrelease import RootReleaseTrainingEngine


class SelectedHeadTrainingEngine(RootReleaseTrainingEngine):
    def _identity(self):
        if self.model.args.model_type not in ('qwen2', 'qwen3'):
            raise ValueError('Selected head supports only verified Qwen2/Qwen3 interfaces')
        value = super()._identity()
        value['schedule'] = 'supervised-position-head-grouped-adam-v1'
        return value

    def _batch(self, rows):
        batch = super()._batch(rows)
        width = batch[0].shape[1]
        selected = [row_index * width + position
                    for row_index, row in enumerate(rows)
                    for position in range(len(row) - 1)
                    if position + 1 >= getattr(row, 'loss_start', 1)]
        if not selected:
            raise ValueError('No supervised target positions')
        return (*batch, self.mx.array(selected, dtype=self.mx.int32))

    def _loss_sum(self, model, inputs, targets, mask, indices):
        hidden = model.model(inputs)
        selected = self.mx.take(hidden.reshape(-1, hidden.shape[-1]), indices, axis=0)
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(selected)
        else:
            logits = model.lm_head(selected)
        selected_targets = self.mx.take(targets.reshape(-1), indices, axis=0)
        return self.nn.losses.cross_entropy(logits.astype(self.mx.float32), selected_targets, reduction='sum')
