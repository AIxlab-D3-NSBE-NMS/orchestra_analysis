import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
import numpy as np
import seaborn as sns

# Quantos participantes é que conseguiram fazer tudo (submeter e terminar a experiência)

# Quantos participantes conseguiram interagir com cycle6 mas não conseguiram submeter

# Quantos participantes começaram o survey mas não conseguiram fazer login na plataforma, separados pelas várias razões (bad gateway, erro 504, problemas na blockchain, falta de certificado, etc.)

# E quantos participantes cancelamos!

# Read the CSV file into a DataFrame
df = pd.read_csv(r'cyclesix_owl.csv', sep=',')

# isolate only valid ROIs
df = df[~df['ROI'].isna().values]

fig_boxplot_durations = plt.figure()
sns.boxplot(data=df, y='duration')
fig_boxplot_durations.tight_layout()
plt.show()

sns.histplot(data=df, x='duration', bins=10)
plt.show()


breakpoint()
