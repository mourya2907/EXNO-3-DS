## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

### CODING AND OUTPUT :
~~~
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('data.csv')
df
~~~
<img width="382" height="270" alt="492752067-e6b162c4-6f42-41f8-9e9a-a680551008fd" src="https://github.com/user-attachments/assets/deb78c03-bc3b-4b57-8e6b-40386b073598" />


~~~
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
climate = ['Cold','Warm','Hot','Very Hot']
ele = OrdinalEncoder(categories=[climate])
ele.fit_transform(df[["Ord_1"]])
~~~
<img width="201" height="167" alt="492752275-5842e538-1477-4ec3-872a-0a0422dadbc2" src="https://github.com/user-attachments/assets/ebb1b969-d835-4560-bab6-0bc53bdff826" />


~~~
df['bo2'] = ele.fit_transform(df[["Ord_1"]])
df
~~~
<img width="430" height="258" alt="492752330-2369811f-ce86-4391-8fe3-240081ddaab6" src="https://github.com/user-attachments/assets/49943b8c-b048-4e6e-9107-6a89d14b066d" />


~~~
le = LabelEncoder()
df2 = df.copy()
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
~~~
<img width="397" height="267" alt="492752619-7ab16cd3-9532-49ca-9420-530725499258" src="https://github.com/user-attachments/assets/0b265716-2142-4122-9546-b4474faacd50" />


~~~
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
~~~
<img width="376" height="252" alt="492752891-a857019d-1a77-4ae0-adb4-53e8a798171d" src="https://github.com/user-attachments/assets/0d88a696-1b8d-48ed-a4b4-44d6a6621793" />


~~~
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
df3 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["City"]]))
df2 = pd.concat([enc,df3],axis = 1)
df2
~~~
<img width="682" height="256" alt="492752907-83d01326-ff11-4edd-bc63-2a1165a9f6fe" src="https://github.com/user-attachments/assets/3942166c-52a3-45a9-aaba-e7e342e1aa30" />


~~~
pd.get_dummies(df,columns=['City'])
~~~
<img width="658" height="291" alt="492753009-6175fc19-9935-40c9-b1dd-86850f0b2eb7" src="https://github.com/user-attachments/assets/5cf954a1-e452-41e7-b3f9-7eebf56f9e49" />


~~~
from category_encoders import BinaryEncoder
df = pd.read_csv('data.csv')
df
~~~
<img width="362" height="261" alt="492753222-895c1abe-b74b-4742-9c7a-947bc6e28f34" src="https://github.com/user-attachments/assets/5709ce9c-b27c-492e-a8b9-9fc32196c3ee" />


~~~
be = BinaryEncoder()
nd = be.fit_transform(df['Ord_2'])
df
~~~
<img width="367" height="257" alt="492753432-1bc31ade-1dbe-4e7b-aac3-4f9c5070f6d2" src="https://github.com/user-attachments/assets/e1495e45-d678-465f-b050-2f1d09222425" />


~~~
from category_encoders import TargetEncoder
te = TargetEncoder()
CC = df.copy()
new = te.fit_transform(CC["City"],y=CC["Target"])
CC = pd.concat([CC,new],axis = 1)
CC
~~~
<img width="418" height="257" alt="492753599-6894a913-cdd9-46e3-8a17-25bb89dffe9d" src="https://github.com/user-attachments/assets/788eea61-a42e-4ba0-86e0-f746e7839f04" />


~~~
if 'City' in CC.columns:
    CC = CC.drop('City', axis=1)
new = te.fit_transform(X = df["City"],y=df["Target"])
CC = pd.concat([CC.reset_index(drop=True),new.reset_index(drop=True)],axis = 1)
CC
~~~
<img width="352" height="265" alt="492753899-8b141315-d5d6-4dbd-ae4b-24cf8dc943d5" src="https://github.com/user-attachments/assets/5e496bbb-fd43-4f04-bbd8-cc2018d62e8b" />


~~~
df = pd.read_csv('Data_to_Transform.csv')
df
~~~
<img width="581" height="310" alt="492754076-fdfd64f1-ec16-4b43-ba88-20293ab157d8" src="https://github.com/user-attachments/assets/34d09467-2967-41b6-8208-c5373b6adc2b" />


~~~
df.skew()
~~~
<img width="247" height="158" alt="492754286-a6670e8f-f743-4530-8ec0-0f40f20cec83" src="https://github.com/user-attachments/assets/60f15fe3-256d-4d5b-80f5-cdf470eac4d0" />


~~~
np.log(df["Highly Positive Skew"])
~~~
<img width="442" height="192" alt="492761806-bddd3395-d66c-4be1-b2ca-0249cfe938ac" src="https://github.com/user-attachments/assets/c4683439-7c8e-4784-9dca-86fcea587a6a" />


~~~
np.reciprocal(df["Moderate Positive Skew"])
~~~
<img width="442" height="201" alt="492761980-dca2a837-0423-4c0f-aad3-dc3687dad4f6" src="https://github.com/user-attachments/assets/616205b8-9c97-425a-891f-db8b73284eb5" />


~~~
np.sqrt(df["Highly Positive Skew"])
~~~
<img width="433" height="207" alt="492762054-3b4a04d3-2e82-4078-beb8-f294623a9911" src="https://github.com/user-attachments/assets/cc295c60-35d6-4d46-ba05-db35d3b8079e" />


~~~
np.square(df["Highly Positive Skew"])
~~~
<img width="431" height="201" alt="492762263-9ef2a921-094a-46eb-bf73-64690196ccd8" src="https://github.com/user-attachments/assets/293e6610-ed20-4f63-aa0c-3062e0bc6742" />


~~~
df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(df["Highly Positive Skew"])
df
~~~
<img width="737" height="338" alt="492762431-e5565716-2960-4582-b601-e45f57a4a990" src="https://github.com/user-attachments/assets/2f5a6f50-b1a0-447c-a0f4-f29ecc3b7385" />


~~~
df["Moderate Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Moderate Negative Skew"])
df
~~~
<img width="753" height="361" alt="492762729-dc7b7359-4049-4c4f-9d23-28c30c8379bb" src="https://github.com/user-attachments/assets/e71d4f8e-eb4f-4f04-a768-29cd921464d9" />


~~~
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
~~~
<img width="745" height="343" alt="492762895-a404a2ec-996e-4493-99f1-51ef8e8f4a70" src="https://github.com/user-attachments/assets/9ba8eb1d-8503-451a-80c6-2f0641163b8b" />


~~~
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line = '45')
plt.show()
~~~
<img width="579" height="432" alt="492762962-de87456b-5a3c-4c31-bf2b-0522b5809a6b" src="https://github.com/user-attachments/assets/15cf1090-7a1a-4de2-b7ee-a72565fc29bc" />


~~~
sm.qqplot(df["Moderate Negative Skew_1"],line = '45')
plt.show()
~~~
<img width="565" height="432" alt="492763042-420705a5-6f3c-4bd8-ac35-fd6716d38caa" src="https://github.com/user-attachments/assets/908e3922-d838-4452-95aa-3ff44afbf389" />


~~~
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line = '45')
plt.show()
~~~
<img width="565" height="432" alt="492763141-d3b0facb-64a4-4ede-af48-1eb37f412654" src="https://github.com/user-attachments/assets/ddf212a7-e7ce-4dd4-b684-e182493d52cc" />


~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line = '45')
plt.show()
~~~
<img width="601" height="432" alt="492763220-324358dc-8ebe-484d-a9f6-47413a345238" src="https://github.com/user-attachments/assets/97ea5035-7b19-476a-8349-bf120c6e9d2b" />


~~~
sm.qqplot(df["Highly Negative Skew_1"],line = '45')
plt.show()
~~~
<img width="565" height="432" alt="492763313-2eadc7fa-e1c0-4007-b1c2-03726d05a36f" src="https://github.com/user-attachments/assets/4efab848-6ba3-4e1d-b3cf-c0d0de297a97" />


~~~
sm.qqplot(np.log(df["Highly Negative Skew_1"]),line = '45')
plt.show()
~~~
<img width="565" height="434" alt="492763397-d40d0617-f287-4149-9afd-5376ef030ec7" src="https://github.com/user-attachments/assets/58c1e45b-1e5d-4ce9-b7bc-edb3fcbc2bc7" />


~~~
sm.qqplot(np.sqrt(df["Moderate Negative Skew_1"]),line='45')
plt.show()
~~~
<img width="565" height="434" alt="492763397-d40d0617-f287-4149-9afd-5376ef030ec7" src="https://github.com/user-attachments/assets/d5827cf3-6ee9-4be1-98e6-53388ca5ecbd" />



~~~
sm.qqplot(np.sqrt(df["Moderate Negative Skew_1"]),line='45')
plt.show()
~~~
<img width="565" height="432" alt="492763453-d3e948f0-6364-4767-ae48-ecae548694ed" src="https://github.com/user-attachments/assets/58128878-f69c-4a0d-9664-3a0f04aafe70" />


~~~
pd.concat([CC,new],axis = 1)
~~~
<img width="418" height="266" alt="492763578-b5535ca2-d372-4b05-9741-b8adb4c6049c" src="https://github.com/user-attachments/assets/8590eca1-c228-47ff-95d8-48bdea2bf7c2" />


### RESULT :
            Thus, we have successfully performed Feature Encoding and Transformation process and saved the data to a file.

















       
