from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


raw_data = datasets.load_wine()

for key, value in raw_data.items():
    print(key, '\n', value, '\n')

print('data.shape\t', raw_data['data'].shape,
      '\ntarget.shape \t', raw_data['target'].shape)

features = pd.DataFrame(data=raw_data['data'], columns=raw_data['feature_names'])
data = features
data['target'] = raw_data['target']
data['class'] = data['target'].map(lambda ind: raw_data['target_names'][ind])

for i in data.target.unique():
    sns.distplot(data['malic_acid'][data.target == i],
                 kde=1, label='{}'.format(i))

plt.legend()
plt.show()


for i in data.target.unique():
    sns.distplot(data['malic_acid'][data.target == i],
                 kde=False, label='{}'.format(i))

plt.legend()
plt.show()

ax = sns.scatterplot(x='alcohol', y='malic_acid', data=data, palette='muted', hue=data.target)
plt.show()

