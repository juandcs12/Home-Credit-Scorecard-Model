# Credit Risk Prediction using Machine Learning
Project Overview
This project focuses on building a credit risk prediction model to estimate the probability of customer default using historical loan data from Home Credit Indonesia. The model is designed to support better decision-making in loan approvals and risk management.

Objectives
- Develop a classification model to predict customer default
- Identify key factors influencing credit risk
- Provide actionable insights for business decision-making

Dataset
The dataset consists of multiple related tables, including:
- Application data (train & test)
- Previous applications
- Credit bureau records
- Installment payments
- Credit card and POS cash balance

Methodology
The project follows a structured data science workflow:
1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering (data aggregation & transformation)
4. Model Building:
   - Logistic Regression
   - XGBoost
   - Random Forest
5. Model Evaluation using ROC-AUC

Key Insights

1. Employment Status Matters
Customers without stable employment (e.g., unemployed) show significantly higher default rates compared to working individuals. This indicates that job stability is a critical factor in assessing credit risk.

2. Education Level Correlates with Risk
Customers with lower education levels tend to have higher default rates, while those with higher education (e.g., academic degree) show lower risk. This suggests a relationship between financial stability and education level.

3. Car Ownership as a Stability Indicator
Customers who own a car tend to have slightly lower default rates, indicating that asset ownership may reflect better financial capability.

4. Regional Risk Variation
Default rates vary across region ratings, with higher-rated regions showing increased default risk. This highlights the importance of geographic segmentation in risk analysis.

5. Weak Indicators
Some variables, such as employee phone ownership, do not show a strong or consistent relationship with default risk and may be less impactful when used individually.

Business Recommendations
- Apply stricter credit evaluation for high-risk groups (e.g., unemployed, lower education level)
- Incorporate multiple factors (not single variables) in credit scoring
- Use geographic and demographic segmentation to improve risk assessment
- Consider asset ownership (e.g., car) as an additional supporting indicator

🛠️ Tools & Technologies
- Python (Pandas, NumPy, Scikit-learn)
- Data Visualization (Matplotlib, Seaborn)
- Google Colab


Author
Juan Daniel Christofer Siahaan
