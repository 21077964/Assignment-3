#LIBRARIES IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

######################SIMPLE_MODEL PART###
# Define the exponential growth model


def exp_growth(x, a, b):
    return a * np.exp(b * x)


# Read the data
data = pd.read_csv("data.csv")
x = data["year"]
y = data["GDP per capita"]

# Fit the model to the data
popt, pcov = curve_fit(exp_growth, x, y)

# Generate the x values for the predictions
x_pred = np.linspace(min(x), max(x) + 20, 100)

# Make predictions using the fitted model
y_pred = exp_growth(x_pred, *popt)

# Compute the standard deviation of the fitting parameters
perr = np.sqrt(np.diag(pcov))

# Compute the confidence interval
lower = y_pred - 1.96 * perr[0] * x_pred
upper = y_pred + 1.96 * perr[0] * x_pred
# Create the plot
plt.plot(x, y, 'o', label='Data')
plt.plot(x_pred, y_pred, '-', label='Best-fitting curve')
plt.fill_between(x_pred, lower, upper, color='gray',
                 alpha=0.2, label='Confidence interval')
plt.legend()
plt.show()



######CO2 VS GDP PART######

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:47:34 2023

@author: addis
"""

import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv("data.csv")

# Create the scatter plot
plt.scatter(data["GDP per capita"], data["CO2 per $ of GDP"])

# Add labels and a title
plt.xlabel("GDP per capita")
plt.ylabel("CO2 emissions per capita")
plt.title("Relationship between GDP per capita and CO2 emissions per capita")

# Show the plot
plt.show()







###################WORLD BANK PART####################
data = pd.read_csv("data.csv")
columns_to_keep = ["GDP per capita", "CO2 production per head",
                   "CO2 per $ of GDP"]
data = data[columns_to_keep]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
kmeans = KMeans(n_clusters=5)
kmeans.fit(data_scaled)
data["cluster"] = kmeans.labels_

cluster_0 = data[data["cluster"] == 0]
cluster_1 = data[data["cluster"] == 1]
cluster_2 = data[data["cluster"] == 2]
cluster_3 = data[data["cluster"] == 3]
cluster_4 = data[data["cluster"] == 4]
# Scatter plot of the clusters
plt.scatter(cluster_0["GDP per capita"],
            cluster_0["CO2 production per head"], c='red', label='cluster 0')
plt.scatter(cluster_1["GDP per capita"],
            cluster_1["CO2 production per head"], c='blue', label='cluster 1')
plt.scatter(cluster_2["GDP per capita"],
            cluster_2["CO2 production per head"], c='green', label='cluster 2')
plt.scatter(cluster_3["GDP per capita"],
            cluster_3["CO2 production per head"], c='purple', label='cluster 3')
plt.scatter(cluster_4["GDP per capita"],
            cluster_4["CO2 production per head"], c='orange', label='cluster 4')

# Plot the cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], c='black', label='Cluster Center')

# Add axis labels and title
plt.xlabel("GDP per capita")
plt.ylabel("CO2 production per head")
plt.title("Cluster Membership and Centers")

# Add a legend
plt.legend()

# Display the plot
plt.show()







