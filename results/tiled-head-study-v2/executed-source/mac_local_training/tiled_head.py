"""Exploratory tile-preserving supervised-position projection.

Pad sparse vocabulary projection to 16 rows to avoid the single-row path.
This is an empirical MLX implementation hypothesis, not a general numerical
proof or a claim of inventing sparse supervision. Dense/high-density batches
use the unchanged dense loss.
"""
from .selected_head import SelectedHeadTrainingEngine
from .engine import TrainingEngine,_json_write

class TiledHeadTrainingEngine(SelectedHeadTrainingEngine):
    def _identity(self):
        value=super()._identity();value['schedule']='supervised-position-head-tile16-grouped-adam-v1';return value

    def _loss_sum(self,model,inputs,targets,mask,indices):
        k=indices.shape[0];padded=((k+15)//16)*16
        if padded>=inputs.size:
            return TrainingEngine._loss_sum(self,model,inputs,targets,mask)
        hidden=model.model(inputs)
        selected=self.mx.take(hidden.reshape(-1,hidden.shape[-1]),indices,axis=0)
        if padded>k:selected=self.mx.concatenate([selected,self.mx.zeros((padded-k,hidden.shape[-1]),dtype=hidden.dtype)])
        logits=model.model.embed_tokens.as_linear(selected) if model.args.tie_word_embeddings else model.lm_head(selected)
        target=self.mx.take(targets.reshape(-1),indices,axis=0)
        return self.nn.losses.cross_entropy(logits[:k].astype(self.mx.float32),target,reduction='sum')

    def run(self):
        report=super().run();report['strategy']='supervised_position_head_tile16_grouped_adam'
        report['limitations'].append('Sparse vocabulary projection is padded to 16 rows; dense/high-density losses retain the dense path. Numerical equivalence is workload- and runtime-specific and requires auditing.')
        _json_write(self.output/'report.json',report);return report
