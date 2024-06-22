import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('/Users/iriszhang/Desktop/messages_history.csv')
# df = df.sample(frac=0.1, random_state=42) 

print(data.head())

# Create contexted conversations
contexted = []
n = 7

for i in range(n, len(data)):
    row = []
    prev = i - 1 - n
    for j in range(i, prev, -1):
        row.append(data.iloc[j]['content'])
    if len(row) == n + 1:  # Ensure we have exactly n+1 elements
        contexted.append(row)

columns = ['response']
columns += ['context/' + str(i) for i in range(n)]

contexted_df = pd.DataFrame.from_records(contexted, columns=columns)

# Split into training and validation datasets
train_df, val_df = train_test_split(contexted_df, test_size=0.1)

# Save to CSV
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)