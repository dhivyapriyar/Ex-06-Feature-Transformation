# Ex-06-Feature-Transformation
AIM
To read the given data and perform Feature Transformation process and save the data to a file.

ALGORITHM
STEP 1
Read the given Data

STEP 2
Clean the Data Set using Data Cleaning Process

STEP 3
Apply Feature Transformation techniques to all the feature of the data set

STEP 4
Save the data to the file

CODE

NAME: DHIVYAPRIYA. R

REG.NO: 212222230032

import pandas as pd

df=pd.read_csv('/content/Data_to_Transform.csv')

df.head()

df.isnull().sum()

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

df['Highly Positive Skew']=np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

df['Highly Positive Skew']=1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

df['Highly Positive Skew']=np.sqrt(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')

plt.show()

from sklearn.preprocessing import PowerTransformer pt=PowerTransformer("yeo-johnson")

df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df['Moderate Negative Skew']=pd.DataFrame(pt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')

plt.show()
OUTPUT
![Screenshot from 2023-05-08 21-35-13](https://user-images.githubusercontent.com/119477552/236875920-6f257433-a345-48e7-951b-9ac4e2ea1a75.png)
![Screenshot from 2023-05-08 21-39-12](https://user-images.githubusercontent.com/119477552/236875990-ae5762da-6d0d-4925-917d-8051340f14c0.png)
![Screenshot from 2023-05-08 21-39-43](https://user-images.githubusercontent.com/119477552/236876050-dc001fc9-30c0-4963-a3fe-352670b329d1.png)
![Screenshot from 2023-05-08 21-39-59](https://user-images.githubusercontent.com/119477552/236876107-1c6c17ec-56de-4406-b76e-312ef8c26e1b.png)
![Screenshot from 2023-05-08 21-40-23](https://user-images.githubusercontent.com/119477552/236876157-cf1b2439-79a2-4c34-8c15-2f7ef90c85a7.png)
![Screenshot from 2023-05-08 21-40-30](https://user-images.githubusercontent.com/119477552/236876369-d67faaa5-26a7-43d9-9556-af86c0ae2fd0.png)
![Screenshot from 2023-05-08 21-40-41](https://user-images.githubusercontent.com/119477552/236876787-45c38ab6-5e35-4ef3-a3fe-90e04f7f8e65.png)
![Screenshot from 2023-05-08 21-40-51](https://user-images.githubusercontent.com/119477552/236876823-16223b1e-70f5-4af7-9017-3ecc5b1dbd37.png)
![Screenshot from 2023-05-08 21-41-01](https://user-images.githubusercontent.com/119477552/236877014-1f1df31d-102e-4eb0-90c2-54a877b304d1.png)
![Screenshot from 2023-05-08 21-41-14](https://user-images.githubusercontent.com/119477552/236877070-fe953bc0-26df-4eaa-8b68-e8b1478b4012.png)
![Screenshot from 2023-05-08 21-41-45](https://user-images.githubusercontent.com/119477552/236877115-35b3b95e-504a-42ed-8535-4b304d9903cd.png)

