import pandas as pd
import matplotlib.pyplot as plt
from figutils import *
from os import path, getlogin, pardir

current_dir = path.dirname(__file__)
data_dir = path.join(current_dir, 'raw_results')
results_dir = path.join(current_dir, pardir, 'figs', 'chapter_2')
filename = 'pubs'

data = pd.read_csv(path.join(data_dir, 'publications.txt'), skiprows=3)

fig, ax = plt.subplots(figsize=set_size(fraction=0.6), layout="constrained")

c = METHOD_PROPS['TF']['color']
data.groupby(data["Publication Year"])["Title"].count().plot(kind="bar", ax=ax, color=c)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel("Publication year") # Sentence case
ax.set_ylim([0,70])

plt.margins(0,0)
fig.savefig(path.join(results_dir, f'{filename}.pdf'),
            transparent=False, dpi=300)#, bbox_inches="tight", pad_inches=0.01)
