import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

# Load your CSV file
df = pd.read_csv('4dme_mean.csv')

# Gender distribution
gender_counts = df['gender (0 or 1)'].value_counts()
print("Gender Distribution:\n", gender_counts)

# Age distribution
age_stats = df['age_mivolo'].describe()
age_skew = skew(df['age_mivolo'].dropna())
print("\nAge Statistics:\n", age_stats)
print("\nAge Skewness:", age_skew)

# Plotting
plt.figure(figsize=(12, 5))

# Histogram for age distribution
plt.subplot(1, 2, 1)
df['age_mivolo'].hist(bins=20)
plt.title('Age Distribution')

# Bar chart for gender distribution
plt.subplot(1, 2, 2)
gender_counts.plot(kind='bar')
plt.title('Gender Distribution')

plt.show()
