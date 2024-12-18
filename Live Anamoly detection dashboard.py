#!/usr/bin/env python
# coding: utf-8

# In[12]:


pip install streamlit


# In[13]:


pip install matplotlib 


# In[10]:


import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import streamlit as st
import time 
import matplotlib.pyplot as plt


# In[29]:


def load_data():
    cashback_claims = pd.read_csv('cashback_claims.csv')
    processed_analysis = pd.read_csv('processed_cashback_analysis.csv')
    return cashback_claims,processed_analysis

cashback_claims,processed_analysis = load_data()


# In[30]:


print(cashback_claims,processed_analysis)


# In[27]:


def preprocess_data(claims, analysis):
    data=pd.concat([claims, analysis],axis=0, ignore_index=True)
    return data
data = preprocess_data(cashback_claims, processed_analysis)
data


# In[28]:


pip install seaborn


# In[22]:


import seaborn as sns


# In[36]:


def anomaly_detection(data):
    model = IsolationForest(contamination=0.05, random_state=42)
    numeric_features = data.select_dtypes(include=['float64', 'int64']).drop(columns=['User ID'], errors='ignore')
    data['anomaly_score'] = model.fit_predict(numeric_features)
    data['status'] = data['anomaly_score'].apply(lambda x: 'Normal' if x == 1 else 'Fraud')
    return data

anomaly_detection(data)



# In[ ]:





# In[41]:


get_ipython().run_cell_magic('writefile', 'visual.py', '\nimport streamlit as st\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\ndef plot_visualizations(data):  # Accept \'data\' as a parameter\n    st.title("Cashback Claims Analysis Dashboard")\n\n    # Status Count Bar Chart\n    st.subheader("Bar Chart: Normal vs Fraudulent Claims")\n    status_counts = data[\'status\'].value_counts()\n    fig, ax = plt.subplots()\n    sns.barplot(x=status_counts.index, y=status_counts.values, palette=\'coolwarm\', ax=ax)\n    ax.set_title("Number of Normal vs Fraudulent Claims")\n    ax.set_xlabel("Status")\n    ax.set_ylabel("Count")\n    st.pyplot(fig)\n\n    # Pie Chart of Status\n    st.subheader("Pie Chart: Proportion of Fraudulent and Normal Claims")\n    fig2, ax2 = plt.subplots()\n    ax2.pie(status_counts, labels=status_counts.index, autopct=\'%1.1f%%\', colors=[\'#4CAF50\', \'#FF5733\'])\n    ax2.set_title("Proportion of Normal vs Fraudulent Claims")\n    st.pyplot(fig2)\n\n    # Scatter Plot for Anomaly Detection\n    st.subheader("Scatter Plot: Anomaly Detection")\n    fig3, ax3 = plt.subplots()\n    numeric_features = data.select_dtypes(include=[\'float64\', \'int64\'])\n    if len(numeric_features.columns) > 1:\n        scatter_x = numeric_features.columns[0]\n        scatter_y = numeric_features.columns[1]\n        sns.scatterplot(x=numeric_features[scatter_x], y=numeric_features[scatter_y], hue=data[\'status\'], palette=\'coolwarm\', ax=ax3)\n        ax3.set_title("Scatter Plot of Anomaly Detection")\n        ax3.set_xlabel(scatter_x)\n        ax3.set_ylabel(scatter_y)\n        st.pyplot(fig3)\n    else:\n        st.warning("Not enough numeric features for a scatter plot.")\n\n# Main Execution\nif __name__ == "__main__":\n    \n    \n    def load_data():\n        data = pd.DataFrame({\n            \'status\': [\'Normal\', \'Fraud\', \'Normal\', \'Fraud\', \'Normal\'],\n            \'feature1\': [1, 5, 3, 8, 2],\n            \'feature2\': [10, 20, 15, 25, 12]\n        })\n        return data\n\n    data = load_data()  # Load your data\n    plot_visualizations(data)  # Pass \'data\' explicitly to the function\n    st.subheader("Processed Data Preview")\n    st.dataframe(data)\n')


# In[ ]:




