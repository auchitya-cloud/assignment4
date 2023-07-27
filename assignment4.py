import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(url, names=names)

# Print the first few rows of the dataset to get an overview
print(df.head())

# Summary statistics of the dataset
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check the number of instances for each class
print(df["class"].value_counts())

# Pairplot to visualize relationships between features
sns.pairplot(df, hue="class")
plt.show()

# Boxplot to visualize the distribution of each feature for each class
plt.figure(figsize=(10, 6))
for i, feature in enumerate(names[:-1]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x="class", y=feature, data=df)
plt.show()

# Correlation heatmap to check the correlation between features
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
