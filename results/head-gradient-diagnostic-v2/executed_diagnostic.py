import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path('outputs/mac-local-training').resolve()))
from mac_local_training.engine import TrainingConfig,TrainingEngine
from mac_local_training.selected_head import SelectedHeadTrainingEngine
p=json.loads(Path('work/reports/selected-head-study-v1/protocol.json').read_text())
out=Path('work/reports/head-gradient-diagnostic-v2');out.mkdir(exist_ok=False)
cfg=TrainingConfig(train_path=p['train']['path'],valid_path=p['valid']['path'],output_path=str(out/'engine'),model_path=p['model_path'],strategy='rotating',steps=1,rank=4,seed=17,**dict(p['config'],window_layers=28))
e=SelectedHeadTrainingEngine(cfg);e.activate_group(0);batch=e._batch(e.train_data[:1]);mx=e.mx

def tiled(model,inputs,targets,mask,indices,tile):
    hidden=model.model(inputs);selected=mx.take(hidden.reshape(-1,hidden.shape[-1]),indices,axis=0)
    k=selected.shape[0];padding=(-k)%tile
    if padding:selected=mx.concatenate([selected,mx.zeros((padding,hidden.shape[-1]),dtype=hidden.dtype)])
    logits=model.model.embed_tokens.as_linear(selected) if model.args.tie_word_embeddings else model.lm_head(selected)
    return e.nn.losses.cross_entropy(logits[:k].astype(mx.float32),mx.take(targets.reshape(-1),indices,axis=0),reduction='sum')

fn=e.nn.value_and_grad(e.model,lambda model,*args:TrainingEngine._loss_sum(e,model,*args))
loss,grads=fn(e.model,*batch[:3]);mx.eval(loss,grads);ref=dict(e.flatten(grads));records=[]
for tile in (1,16,32,64,128,256):
    fn=e.nn.value_and_grad(e.model,lambda model,*args:tiled(model,*args,tile=tile))
    other,g=fn(e.model,*batch);mx.eval(other,g);flat=dict(e.flatten(g))
    errors={name:float(mx.max(mx.abs(flat[name]-value)).item()) for name,value in ref.items()}
    records.append({'tile':tile,'loss':float(other.item()),'loss_difference':float(other.item()-loss.item()),'max_gradient_error':max(errors.values()),'largest_error_key':max(errors,key=errors.get)})
result={'status':'post-hoc exploratory gradient diagnostic; no confirmation decision','reference_loss':float(loss.item()),'seed':17,'rank':4,'context_tokens':256,'supervised_positions':batch[3].size,'records':records}
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
