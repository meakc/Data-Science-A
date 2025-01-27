# Customer Purchase Behavior Analysis, Sales Prediction Using Random Forest, Customer Segmentation via K-Means Clustering
The provided code is a comprehensive Python script designed to perform data analysis, visualization, and customer segmentation using machine learning techniques. It leverages libraries such as pandas, matplotlib, seaborn, and scikit-learn to process datasets related to customers, products, and transactions. The script is organized into three primary tasks: Data Exploration and Visualization, Lookalike Modeling, and Customer Segmentation.

**1. Data Exploration and Visualization**

In this initial phase, the script focuses on loading and examining the datasets to understand their structure and contents.

- **Loading Libraries and Datasets**: The script begins by installing and importing the necessary libraries:

  ```python
  !pip install pandas matplotlib seaborn
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  

  It then loads the datasets from specified URLs:

  ```python
  customers_url = 'https://drive.google.com/uc?id=1bu_--mo79VdUG9oin4ybfFGRUSXAe-WE'
  products_url = 'https://drive.google.com/uc?id=1IKuDizVapw-hyktwfpoAoaGtHtTNHfd0'
  transactions_url = 'https://drive.google.com/uc?id=1saEqdbBB-vuk2hxoAf4TzDEsykdKlzbF'

  customers = pd.read_csv(customers_url)
  products = pd.read_csv(products_url)
  transactions = pd.read_csv(transactions_url)
  

- **Dataset Overview**: To gain insights into the data, the script prints information and descriptive statistics for each dataset:

  ```python
  print("Customers Data:")
  print(customers.info())
  print(customers.describe())

  print("Products Data:")
  print(products.info())
  print(products.describe())

  print("Transactions Data:")
  print(transactions.info())
  print(transactions.describe())
  

  This step provides an understanding of the data types, missing values, and basic statistical measures.

- *Data Visualization*: The script utilizes seaborn and matplotlib to create visual representations of the data:

  - *Customer Distribution by Region*: A count plot to show the number of customers in each region:

    ```python
    plt.figure(figsize=(10, 6))
    sns.countplot(data=customers, x='Region', palette='viridis')
    plt.title('Customer Distribution by Region')
    plt.show()
    
![Alt text](https://github.com/meakc/Data-Science-A/blob/main/customer_distribution_by_region.png?raw=true)
  - **Product Price Distribution by Category**: A box plot to display the price distribution across different product categories:

    ```python
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=products, x='Category', y='Price', palette='coolwarm')
    plt.title('Product Price Distribution by Category')
    plt.xticks(rotation=45)
    plt.show()
    
![Alt text](https://github.com/meakc/Data-Science-A/blob/main/product_price_distribution_by_catogory.png?raw=true)
  - **Transaction Value Distribution**: A histogram to illustrate the distribution of transaction values:

    ```python
    plt.figure(figsize=(10, 6))
    sns.histplot(transactions['TotalValue'], bins=30, kde=True, color='blue')
    plt.title('Transaction Value Distribution')
    plt.show()
    
![Alt text](https://github.com/meakc/Data-Science-A/blob/main/Transaction_value_distribution.png?raw=true)
  These visualizations help in identifying patterns and anomalies within the data.

- *Insights*: Based on the analysis, the script lists several business insights:

  ```python
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
  

  These insights can inform strategic decisions in marketing and sales.

**2. Lookalike Modeling**

The second task involves identifying customers with similar purchasing behaviors using cosine similarity.

- **Merging Datasets**: The script merges the transactions, customers, and products datasets to create a comprehensive view:

  ```python
  merged = transactions.merge(customers, on='CustomerID', how='left')
  merged = merged.merge(products, on='ProductID', how='left')
  

- *Customer-Product Interaction Matrix*: It then aggregates the data to create a matrix representing the total value of products purchased by each customer:

  ```python
  customer_product_features = merged.groupby(['CustomerID', 'ProductID']).agg(
      {'Quantity': 'sum', 'TotalValue': 'sum'}).reset_index()

  customer_product_matrix = customer_product_features.pivot(index='CustomerID',
                                                            columns='ProductID',
                                                            values='TotalValue').fillna(0)
  

- **Normalization and Similarity Calculation**: The matrix is normalized, and cosine similarity is computed to measure the similarity between customers:

  ```python
  normalized_matrix = customer_product_matrix.div(customer_product_matrix.sum(axis=1), axis=0)

  from sklearn.metrics.pairwise import cosine_similarity
  similarity_scores = cosine_similarity(normalized_matrix)

  similarity_df = pd.DataFrame(similarity_scores, index=customer_product_matrix.index,
                               columns=customer_product_matrix.index)
  

- **Identifying Lookalike Customers**: For the first 20 customers, the script identifies the top three most similar customers:

  ```python
  lookalike_results = {}
  for customer in similarity_df.index[:20]:
      top_matches = similarity_df[customer].nlargest(4).iloc[1:]
      lookalike_results[customer] = [(match, score) for match, score in top_matches.items()]

  lookalike_df = pd.DataFrame.from_dict(lookalike_results, orient='index',
                                        columns=['Lookalike1', 'Lookalike2', 'Lookalike3'])
  lookalike_df.to_csv('Lookalike.csv')

**3: Customer Segmentation**

**Install Necessary Libraries**
Run the following command to install the required libraries:
`!pip install pandas scikit-learn matplotlib seaborn`

**Import Required Libraries**
The script uses the following libraries:
- `pandas` for data manipulation
- `scikit-learn` for clustering and scaling
- `matplotlib` and `seaborn` for visualization

**Load Datasets**
Datasets are loaded using the URLs provided:
- `customers_url`: [Customers Dataset](https://drive.google.com/uc?id=1bu_--mo79VdUG9oin4ybfFGRUSXAe-WE)
- `products_url`: [Products Dataset](https://drive.google.com/uc?id=1IKuDizVapw-hyktwfpoAoaGtHtTNHfd0)
- `transactions_url`: [Transactions Dataset](https://drive.google.com/uc?id=1saEqdbBB-vuk2hxoAf4TzDEsykdKlzbF)

These datasets are read into DataFrames using `pandas.read_csv`.

**Merge Datasets and Verify**
The `transactions`, `customers`, and `products` datasets are merged into a single DataFrame to consolidate all relevant information. If the `Price` column is missing or entirely null, a default value of `0.0` is assigned.

**Feature Selection for Clustering**
Key features used for clustering include:
- `TotalValue`: The total value of transactions per customer
- `Quantity`: The total quantity of items purchased
- `Price`: The average price of products purchased

**Normalize Data**
The features are normalized using `StandardScaler` to ensure they are on the same scale.

**Perform K-Means Clustering**
K-Means clustering is applied with 4 clusters. Each customer is assigned to a cluster, which is added as a new column in the dataset.

**Evaluate Clusters**
The quality of the clustering is evaluated using the Davies-Bouldin Index, which provides a measure of cluster separation and compactness.

**Visualize Clusters**
A scatter plot is generated to visualize the customer clusters in a 2D space using scaled features.

**Save Clustering Results**
The clustering results, including the cluster assignments, are saved to a file named `Customer_Segmentation.csv`.

**Outputs**:
1. **Davies-Bouldin Index**: Evaluates the quality of the clustering model.
2. **Visualization**: Displays customer clusters in a scatter plot.
3. **Saved Results**: The segmentation details are saved in `Customer_Segmentation.csv`.
![Alt text](https://github.com/meakc/Data-Science-A/blob/main/customer_segmentation_clusture.png?raw=true)


# BY ABHISHEK KUMAR CHOUDHARY
