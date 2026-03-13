import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Quantos participantes é que conseguiram fazer tudo (submeter e terminar a experiência)

# Quantos participantes conseguiram interagir com cycle6 mas não conseguiram submeter

# Quantos participantes começaram o survey mas não conseguiram fazer login na plataforma, separados pelas várias razões (bad gateway, erro 504, problemas na blockchain, falta de certificado, etc.)

# E quantos participantes cancelamos!

# Read the CSV file into a DataFrame
df = pd.read_csv(r'cyclesix.csv', sep='\t')

# Create a new column 'submitted' with boolean values


outcomes = {'completo': 0, # status online, submitted True e exclude not found in comments
            'nosubmit': 0, # was online and coun't submit
            'offline': 0}

# Count the number of offline entries
offline_count = df[~df['status_website'].isin(['online'])].shape[0]
outcomes['offline'] = offline_count
outcomes['nosubmit'] = int(np.count_nonzero(np.logical_and( df.status_website=='online',
                                                            df.submitted==0,
                                                            ~df['comments'].str.contains('exclude', na=False))))
outcomes['completo'] = int(np.count_nonzero((np.logical_and( df.status_website=='online',
                                                            df.submitted==1,
                                                            ~df['comments'].str.contains('exclude', na=False)))))

print(outcomes)

completed_idx = np.logical_and( df.status_website=='online',
                                df.submitted==1,
                                ~df['comments'].str.contains('exclude', na=False))
df_completed = df[completed_idx]
df_online = df[df['status_website'].isin(['online'])]
df_online['submitted'] = df_online['submitted'].astype(bool)



sns.histplot(data=df_online, x='submitted', bins='auto', kde=False, color='k', discrete=True)
plt.title('Freq. de sucesso na submissão')
plt.xlabel('Submetido')
plt.ylabel('Frequência')
plt.xticks([0,1], ['0', '1'])
plt.show()

# Plot a histogram of 'tentativas' for all its unique valu



breakpoint()




df['submitted'] = df['submitted'].astype(bool)

# Group the data by 'readable_date' and 'submitted'
grouped_df = df.groupby(['readable_date', 'submitted']).size().reset_index(name='count')

# Convert 'readable_date' to string format for plotting
grouped_df['readable_date'] = grouped_df['readable_date'].astype(str)

# Count the number of offline entries
offline_count = df[~df['status_website'].isin(['online'])].shape[0]
print(f"Number of offline times: {offline_count}")

# Print website was offline at specific times if there are any
if not grouped_df[grouped_df['submitted'] == False].empty:
    print("Website was offline at the following times:")
    for index, row in grouped_df[grouped_df['submitted'] == False].iterrows():
        print(f"Date: {row['readable_date']}, Time: {df[df['readable_date'] == row['readable_date']]['readable_time'].iloc[0]}")

# Create a stacked barplot using matplotlib
plt.figure(figsize=(12, 8))
dates = grouped_df['readable_date'].unique()
submitted_counts = grouped_df[grouped_df['submitted']].groupby('readable_date')['count'].sum().reindex(dates)
not_submitted_counts = grouped_df[~grouped_df['submitted']].groupby('readable_date')['count'].sum().reindex(dates)

plt.bar(dates, submitted_counts, label='Submitted')
plt.bar(dates, not_submitted_counts, bottom=submitted_counts, label='Not Submitted')
















plt.title('Submitted vs. Not Submitted Entries per Day')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Submitted/Not Submitted')
plt.show()
