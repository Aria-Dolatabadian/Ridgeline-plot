import pandas as pd
import matplotlib.pyplot as plt
from joypy import joyplot

# Load dataset
data = pd.read_csv('probly.csv')

# Data transformation
data = data.melt(var_name="text", value_name="value")
data["text"] = data["text"].str.replace(".", " ", regex=False)
data["value"] = data["value"].round(0)
filtered_texts = [
    "A", "B", "C", "D",
    "E", "F", "G", "H"
]
data = data[data["text"].isin(filtered_texts)]

# Reorder categories based on the mean value
data["text"] = pd.Categorical(data["text"], ordered=True,
                              categories=data.groupby("text")["value"].mean().sort_values().index)

# Define a color palette with different colors for each category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Ridge plot with custom colors
plt.figure(figsize=(8, 4))
joyplot(
    data=data,
    by="text",
    column="value",
    bins=20,
    grid=True,
    fill=True,
    linewidth=1,
    alpha=0.6,
    figsize=(10, 6),
    color=colors
)

# Customise plot
plt.xlabel("Assigned Probability (%)")
plt.ylabel("Categories")
plt.title("Ridge Plot")
plt.show()
