# Comprehensive Data Analysis, Visualization, and Customer Segmentation Using Python

This repository contains a Python script designed to perform data analysis, visualization, and customer segmentation using advanced machine learning techniques. The script leverages libraries such as pandas, matplotlib, seaborn, and scikit-learn. Below is a detailed overview of the tasks and functionalities covered in the script.

---

## 1. Data Exploration and Visualization

### Overview

This phase focuses on loading, examining, and visualizing datasets related to customers, products, and transactions.

### Loading Libraries and Datasets

The following libraries are used in the script:

python
!pip install pandas matplotlib seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


Datasets are loaded from the specified URLs:

python
customers_url = 'https://drive.google.com/uc?id=1bu_--mo79VdUG9oin4ybfFGRUSXAe-WE'
products_url = 'https://drive.google.com/uc?id=1IKuDizVapw-hyktwfpoAoaGtHtTNHfd0'
transactions_url = 'https://drive.google.com/uc?id=1saEqdbBB-vuk2hxoAf4TzDEsykdKlzbF'

customers = pd.read_csv(customers_url)
products = pd.read_csv(products_url)
transactions = pd.read_csv(transactions_url)


### Dataset Exploration

To understand the structure and content of the datasets:

python
print("Customers Data:")
print(customers.info())
print(customers.describe())

print("Products Data:")
print(products.info())
print(products.describe())

print("Transactions Data:")
print(transactions.info())
print(transactions.describe())


### Data Visualization

Key visualizations include:

- *Customer Distribution by Region*:

  python
  plt.figure(figsize=(10, 6))
  sns.countplot(data=customers, x='Region', palette='viridis')
  plt.title('Customer Distribution by Region')
  plt.show()
  



- *Product Price Distribution by Category*:

  python
  plt.figure(figsize=(10, 6))
  sns.boxplot(data=products, x='Category', y='Price', palette='coolwarm')
  plt.title('Product Price Distribution by Category')
  plt.xticks(rotation=45)
  plt.show()
  



- *Transaction Value Distribution*:

  python
  plt.figure(figsize=(10, 6))
  sns.histplot(transactions['TotalValue'], bins=30, kde=True, color='blue')
  plt.title('Transaction Value Distribution')
  plt.show()
  



### Insights

Based on the data analysis:

python
insights = [
    "1. Customers are primarily distributed across specific regions, with Region X having the highest concentration.",
    "2. Category A products are generally more expensive than others, with a wider price range.",
    "3. Transaction values are right-skewed, dominated by a few high-value transactions.",
    "4. Seasonal trends show peak transactions during specific months.",
    "5. Customer signup dates indicate steady growth in acquisitions."
]

print("Business Insights:")
for insight in insights:
    print(insight)


---

## 2. Lookalike Modeling

### Overview

This phase identifies customers with similar purchasing behaviors using cosine similarity.

### Steps

- *Dataset Merging*:

  python
  merged = transactions.merge(customers, on='CustomerID', how='left')
  merged = merged.merge(products, on='ProductID', how='left')
  

- *Customer-Product Interaction Matrix*:

  python
  customer_product_features = merged.groupby(['CustomerID', 'ProductID']).agg(
      {'Quantity': 'sum', 'TotalValue': 'sum'}).reset_index()

  customer_product_matrix = customer_product_features.pivot(index='CustomerID',
                                                            columns='ProductID',
                                                            values='TotalValue').fillna(0)
  

- *Normalization and Similarity Calculation*:

  python
  normalized_matrix = customer_product_matrix.div(customer_product_matrix.sum(axis=1), axis=0)

  from sklearn.metrics.pairwise import cosine_similarity
  similarity_scores = cosine_similarity(normalized_matrix)

  similarity_df = pd.DataFrame(similarity_scores, index=customer_product_matrix.index,
                               columns=customer_product_matrix.index)
  

- *Identifying Lookalike Customers*:

  python
  lookalike_results = {}
  for customer in similarity_df.index[:20]:
      top_matches = similarity_df[customer].nlargest(4).iloc[1:]
      lookalike_results[customer] = [(match, score) for match, score in top_matches.items()]

  lookalike_df = pd.DataFrame.from_dict(lookalike_results, orient='index',
                                        columns=['Lookalike1', 'Lookalike2', 'Lookalike3'])

  lookalike_df.to_csv('Lookalike.csv')
  

---

## Download the README File

[Download README.md](./README.md)
