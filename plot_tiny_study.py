"""Recreate the post-run diagnostic figure from published JSON; no training."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
r=json.loads((root/'results/tiny-residency-20261011/study.json').read_text());rows={(x['cell']['seed'],x['cell']['strategy']):x['report'] for x in r['results']};seeds=[11,29,47]
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
for axis,key,ylabel in [(axes[0],'peak_optimizer_resident_bytes','Resident optimizer state (bytes)'),(axes[1],'heldout_nll_change','Validation NLL change (nats/token)')]:
 for offset,strategy,color in [(-.18,'all','#2563eb'),(.18,'rotating','#8b5cf6')]:
  axis.bar([i+offset for i in range(3)],[rows[seed,strategy][key] for seed in seeds],width=.34,label=strategy,color=color)
 axis.set_xticks(range(3),[str(seed) for seed in seeds]);axis.set_xlabel('Fixed initialization/data-order seed');axis.set_ylabel(ylabel)
axes[0].legend(frameon=False);axes[1].axhline(0,color='#94a3b8',linewidth=1)
fig.suptitle('Random tiny-model check: optimizer residency and optimization differ',weight='bold',fontsize=13)
fig.text(.01,-.07,'16 updates · same token exposure within each seed · rotating layers receive 4 updates each vs 16 for all\nOptimizer bytes are not total RAM. Synthetic/random models; no useful-model-quality claim. Megern',fontsize=9,color='#475569')
fig.savefig(root/'results/tiny-residency-20261011/residency-and-nll.png',dpi=180,bbox_inches='tight');plt.close(fig)
