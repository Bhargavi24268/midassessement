import requests
from bs4 import BeautifulSoup
import pandas as pd
url = "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Initialize empty lists to store data
titles = []
categories = []
prices = []

# Scrape book titles, prices, and categories
books = soup.find_all('article', class_='product_pod')
for book in books:
    title = book.h3.a['title']
    price = book.find('p', class_='price_color').text.strip().replace('£', '')
    category = 'Travel'  # Fixed category for this page

    titles.append(title)
    prices.append(float(price))
    categories.append(category)
data = pd.DataFrame({'Title': titles, 'Category': categories, 'Price': prices})

# Save to CSV
data.to_csv('books_to_scrape.csv', index=False)
print("Data scraped and saved to books_to_scrape.csv")
# Load the dataset
df = pd.read_csv('books_to_scrape.csv')

# Data cleaning
print(df.info())  
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')  # Ensure prices are numeric
df.dropna(inplace=True)  # Remove rows with missing values
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Price'], bins=10, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Visualize category count
plt.figure(figsize=(8, 5))
sns.countplot(x='Category', data=df)
plt.title("Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Feature and target selection
X = df[['Price']]  # For this example, let's predict price itself (Dummy Task)
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))
