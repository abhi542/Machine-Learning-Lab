import pandas as pd
import numpy as np

data = pd.read_csv("id3 and naive bayes data.csv")

def entropy(col):
    counts = col.value_counts()
    probs = counts / len(col)
    ent = -np.sum(probs * np.log2(probs))
    return ent

def info_gain(df, feat, target="target"):
    total_ent = entropy(df[target])
    vals = df[feat].unique()
    weighted_ent = 0
    for val in vals:
        sub_df = df[df[feat] == val]
        prob = len(sub_df) / len(df)
        weighted_ent += prob * entropy(sub_df[target])
    return total_ent - weighted_ent

def id3(df, orig_df, feats, target="target", parent_class=None):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if len(df) == 0:
        return orig_df[target].mode()[0]
    if not feats:
        return parent_class
    
    parent_class = df[target].mode()[0]
    gains = [info_gain(df, feat, target) for feat in feats]
    best_feat = feats[np.argmax(gains)]
    tree = {best_feat: {}}
    
    feats = [f for f in feats if f != best_feat]
    for val in df[best_feat].unique():
        sub_df = df[df[best_feat] == val]
        subtree = id3(sub_df, orig_df, feats, target, parent_class)
        tree[best_feat][val] = subtree
    
    return tree

feats = list(data.columns[:-1])
target = data.columns[-1]

tree = id3(data, data, feats, target)
print("Decision Tree:", tree)
