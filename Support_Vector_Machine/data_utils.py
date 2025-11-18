import seaborn as sns

def load_data():
    df = sns.load_dataset("tips")
    return df
