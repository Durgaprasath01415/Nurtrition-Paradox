import streamlit as st
import matplotlib.pyplot as plt
import pymysql as sql
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew

#Loading Data
myconnection = sql.connect(host ="localhost",user = "root",password = "Mysql")
cur = myconnection.cursor()
cur.execute("use nutrition")
df = pd.read_sql("select * from nutrition.obesity",con = myconnection)


st.write('''# Univariate Analysis Histogram Plot For Numerical Features''')

numeric_features = ['lower_bound', 'upper_bound', 'mean_estimate', 'ci_width', 'year']
# Histo plot of Numerical columns
fig = plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
for i in range(0, len(numeric_features)):
    plt.subplot(5, 3, i+1)
    sns.histplot(x=df[numeric_features[i]],kde=True, color='g')
    plt.xlabel(numeric_features[i])
    plt.title(f'Distribution Plot of {numeric_features[i]}')
plt.tight_layout()
# Display the plot in Streamlit
st.pyplot(fig)

st.write('''# Obervation
    1. lower_bound, upper_bound, mean_estimate and ci_width distributions are positive skewed or right skewed and it is leptoKurtosis.
    2. lower_bound, upper_bound, mean_estimate and ci_width distributions the central tendency value like  be Mean > Median > Mode.
    3. The year column show the uniform distribution of data.
    4. Year column has no skewness since the distribution is uniform and Mean = Median = Mode.
    5. Most of the value of lower_bound are to the left side of ploting which shows that data points are to the minimum.
    6. Most of the value of upper_bound are to the left side of ploting which shows that data points are minimum.
    7. Most of the value of mean_estimate are to the left side of ploting which shows that data points are minimum.
    8. Most of the value of ci_width are to the left side of ploting which shows that data points are minimum.
    5. On each year the data entries points are equal.''')


st.write('''# Univariate Analysis Box Plot For Numerical Features''')
# Box plot of Numerical columns to see outlier
plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
for i,col in enumerate(numeric_features):
    plt.subplot(5, 3, i+1)
    sns.boxplot(x=df[col], color='lightgreen')
    plt.xlabel(numeric_features[i])
    plt.title(f'Plot of {numeric_features[i]}')
plt.tight_layout()
st.pyplot(plt)
st.write(''' # Obervation
    1) Median line of lower_bound, upper_bound, mean_estimate and ci_width are towards the left side of the plot.
    2) We can confirm from the boxplot that lower_bound, upper_bound, mean_estimate and ci_width are positive skewed or right skewed.
    3) Year has symmetric distribution and Median line is almost in mid of the distribution plot.
    4) lower_bound, upper_bound, mean_estimate and ci_width columns have many outlier data point.''')

#Function to get the number of outlier in the columns
def count_outliers_iqr(data):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return len(outliers)

#Seeing the number of outlier in columns
st.write('# Number of Outlier')
st.write(f'lower_bound : {count_outliers_iqr(df['lower_bound'])}')
st.write(f'upper_bound : {count_outliers_iqr(df['upper_bound'])}')
st.write(f'mean_estimate : {count_outliers_iqr(df['mean_estimate'])}')
st.write(f'ci_width : {count_outliers_iqr(df['ci_width'])}')

# Checking the skewness of the numerical columns
st.write('# Skewness of dataset')
st.write(f'skewness_of_lower_bound : {skew(df['lower_bound'])}')
st.write(f'skewness_of_upper_bound : {skew(df['upper_bound'])}')
st.write(f'skewness_of_mean_estimate : {skew(df['mean_estimate'])}')
st.write(f'skewness_of_ci_width : {skew(df['ci_width'])}')

st.write('# Univariate Analysis Count Plot Categorical Feature')
# Histo plot for Categorical columns
categorical_cols = ['region', 'gender', 'age_group', 'obesity']
plt.figure(figsize=(18, 25))
for i, col in enumerate(categorical_cols):
    plt.subplot(5, 2, i + 1)
    sns.countplot(x=col, data=df, color='lightgreen',order = df[col].value_counts().index)
    plt.title(f'Plot of {col}')
    plt.xticks(rotation=45, ha='right')
plt.suptitle('Count Plots of Categorical Variables',  fontsize=20, fontweight='bold', alpha=0.8, y=1.)
plt.tight_layout()
st.pyplot(plt)
st.write('''# Observations of the categorical variables:

## region:

    1) 'Europe', 'Africa', and 'Americas' are the regions with the highest number of entries in the dataset.
    2) 'Western Pacific', 'Eastern Mediterranean', and 'South-East Asia' have progressively fewer entries.
    3) The 'Unknown' category (imputed values) also known has a nan of entries, which confirms the missing data imputation.

## gender:

    1) The counts for 'Male', 'Female', and 'Both' genders are almost perfectly equal. 
    2) This indicates a balanced representation of gender categories in the dataset.

## age_group:

    1) The 'Child' age group has a higher count compared to the 'Adult' age group.
    2) Since child count is higher, the data of childer's obesity data is more then the adult's in dataset.

## obesity:

    1) The 'Low' obesity category is higher count in the dataset, compare to 'High' and 'Moderate' categories combined.
    2) 'High' obesity has a higher count than 'Moderate' obesity.
    3) 'Moderate' obesity has a leastest count.''')

st.write('# Bivariate Analysis HeatMap Between Numericals Feature')
# 1. Numerical vs. Numerical
# Correlation Matrix and Heatmap
numerical_cols = ['lower_bound', 'upper_bound', 'mean_estimate', 'ci_width', 'year']
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.xticks(rotation=45, ha='right')
plt.title('Correlation Matrix of Numerical Variables')
st.pyplot(plt)

st.write('''# Observations from the correlation matrix

## Strong Positive Correlations (Near 1.0):

    mean_estimate is highly correlated with both lower_bound (0.96) and upper_bound (0.96). This is expected, as mean_estimate is the central value between lower_bound and upper_bound, and these three variables inherently describe the same estimated range.
    lower_bound and upper_bound also show a strong positive correlation with each other (0.85).

## Moderate Positive Correlation (Around mid 0.5):

    ci_width (confidence interval width) shows a moderate positive correlation with upper_bound (0.61) and mean_estimate (0.35). This suggests that as the estimated obesity percentage (mean_estimate) increases, or the upper limit of the estimate (upper_bound) increases, the confidence interval around that estimate tends to get wider. This could indicate greater variability or uncertainty in higher obesity estimates.

## Weak Positive Correlations (Near 0):

    year has a weak positive correlation with ci_width (0.25), upper_bound (0.17), and mean_estimate (0.11). This suggests a slight, albeit not strong, increasing trend in the estimated obesity percentages and the width of their confidence intervals over the years in the dataset.
    lower_bound shows a very weak positive correlation with year (0.04).

## No Strong Negative Correlations:

    There are no strong negative correlations observed between any of the numerical variables, indicating that as one variable increases, the other doesn't consistently decrease.''')

st.write('# Bivariate Analysis Bar Plot Between Categorical and Numericals Features')
# 2. Categorical vs. Numerical
# Box plots of mean_estimate and ci_width across categorical variables
categorical_num_pairs = [
    ('region', 'mean_estimate'),
    ('gender', 'mean_estimate'),
    ('age_group', 'mean_estimate'),
    ('obesity', 'mean_estimate'),
    ('region', 'ci_width'),
    ('gender', 'ci_width'),
    ('age_group', 'ci_width'),
    ('obesity', 'ci_width')
]

plt.figure(figsize=(18, 15))
for i, (cat_col, num_col) in enumerate(categorical_num_pairs):
    plt.subplot(4, 2, i + 1)
    sns.barplot(x=cat_col, y=num_col, data=df)
    plt.title(f'{num_col} by {cat_col}')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.suptitle('Categorical vs. Numerical Variables', y=1.02)
st.pyplot(plt)
st.write('''# Observations from plots 
    1.mean_estimate and region:
        'Western Pacific' and 'Americas' regions show the highest mean_estimate values, indicating high obesity percentages in these regions.
        'South-East Asia' and 'Africa' have low mean_estimate values.
        'Eastern mediterranean' and 'Europe' have moderate mean_estimate values.
        The 'Unknown' region has an mean_estimate somewhere in the middle, reflecting the averages of the countries it represents whose regions were initially missing.

    2.mean_estimate and gender:
        The bars are almost at the same height. So mean_estimate values is similar across 'Male', 'Female', and 'Both' genders.

    3.mean_estimate and age_group:
        This plot shows a large difference, the 'Adult' age group has a significantly higher mean_estimate compared to the 'Child' age group. This clearly indicates that obesity percentages much higher in adults than in children.

    4.mean_estimate and obesity:
        The mean_estimate values are clearly separated by the obesity categories: 'High' has the highest, 'Moderate' is in the middle, and 'Low' has the lowest mean_estimate.

    5.ci_width and region:
        Similar to mean_estimate, 'Americas' and 'Western Pacific'show high level in ci_width values as well.
        'Eastern mediterranean' have moderately lesser level then 'Americas' and 'Western Pacific'.
        'South-East Asia','Europe' and 'Africa' have almost similar ci_width values.
        'Unknown' have the least ci_width values.

    6.ci_width by gender:
        The ci_width values are high in "Male','Female' is slight less then 'Male' and 'Both' in slight less then 'Female'.

    7.ci_width by age_group:
        The 'Child' age group has a noticeably higher ci_width than the 'Adult' age group.

    8.ci_width by obesity:
        The ci_width values are clearly separated by the obesity categories: 'High' has the highest, 'Moderate' is in the middle, and 'Low' has the lowest ci_width.''')

st.write('# Bivariate Analysis Bar Plot Between Categorical Features')
# 3. Categorical vs. Categorical
# Stacked Bar Charts for categorical relationships

st.write('# Region vs Obesity')
df_region_obesity = pd.crosstab(df['region'], df['obesity'], normalize='index')
df_region_obesity.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title('Obesity Distribution by Region')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
st.pyplot(plt)
st.write('''## Observations from the stacked bar 'Obesity Distribution by Region':

    Across all regions, the 'Low' obesity category as the largest proportion.

    'Western Pacific' and 'Americas' show a visibly larger proportion of 'High' and 'Moderate' obesity compared to other regions.

    'Europe' and 'Eastern Mediterranean' also have 'High' and 'Moderate' obesity, but their proportions are smaller than 'Western Pacific' and 'Americas'.

    'Africa' has a very small proportion of 'High' and 'Moderate' obesity, with 'Low' almost the entire bar.

    'South-East Asia' region stands out as it shows almost exclusively 'Low' obesity, with no proportions of 'High' or 'Moderate' obesity.

    'Unknown' Region: The 'Unknown' region (which was imputed) also have 'High' and 'Moderate' obesity, with 'Low' as small proportions.''')

st.write('# Gender vs Obesity')
df_gender_obesity = pd.crosstab(df['gender'], df['obesity'], normalize='index')
df_gender_obesity.plot(kind='bar', stacked=True, figsize=(8, 5))
plt.title('Obesity Distribution by Gender')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
st.pyplot(plt)
st.write('''## Observations from the stacked bar chart of 'Obesity Distribution by Gender':

    For all gender categories ('Male', 'Female', and 'Both'), the 'Low' obesity category is majorly observed.

    'Male', 'Female' and 'Both' categories have almost a similar proportion of 'high' obesity.

    'Female' categories shows, larger 'low' obesity proportion comparing to 'Male' and 'Both'.

    Categories of the gender shows that obesity is not only relating towards one or two gender categories but all gender have the obesity.''')

st.write('# Age Group vs Obesity')
df_agegroup_obesity = pd.crosstab(df['age_group'], df['obesity'], normalize='index')
df_agegroup_obesity.plot(kind='bar', stacked=True, figsize=(8, 5))
plt.title('Obesity Distribution by Age Group')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
st.pyplot(plt)
st.write('''## Observations from the stacked bar chart of 'Obesity Distribution by Age Group':

    This plot reveals a very strong and clear relationship between age_group and the distribution of obesity categories.

    'Child' age_group is majorly of 'Low' Obesity.

    The proportions of 'Moderate' and 'High' obesity among children are extremely small. 
    
    This suggests that, in this dataset, child obesity is relatively uncommon.

    'Adult' age_group is also majorly of 'Low' obesity, but comparing to the 'child' age_group the 'High' and 'Moderate' obesity segments have larger proportion.

    This plot strongly indicates that adults are considerably more to 'Moderate' and 'High' obesity compared to children in this dataset.''')


st.write('''# Summary of the EDA's of obesity dataset

The EDA revealed a dataset rich in demographic(age_group) and geographical(region) information related to obesity

    The majority of observations fall into the 'Low' obesity category. However, the 'High' and 'Moderate' categories are comparatively  smaller, represent crucial areas for intervention and study. This imbalance is a primary concern for predictive modeling.

    Age is the demographic factor:

        Adults show significantly higher obesity estimates and a much greater proportion of 'High' and 'Moderate' obesity compared to 'Children'.
        In univariant analysis even through Number of entries in 'child' age_group is high we are able to see that 'adult' age_group has the significantly higher obesity.
        This suggests that interventions and public health strategies related to obesity should strongly prioritize the adult population.
        Children are to the 'Low' obesity category in this dataset.

    Regional is the geographical factor:

        Western Pacific and Americas regions exhibit the highest obesity estimates and the largest proportions of 'High' and 'Moderate' obesity.These regions likely require attention for obesity prevention and management.
        South-East Asia stands out with almost 'Low' obesity observations

    Gender Shows Minimal Impact on Obesity: 
        Across the all three categories of gender, the proportional distribution of gender categories and obesity are similar for 'Male', 'Female', and 'Both' genders. This indicates that, within this dataset, gender itself is not a strong differentiating factor for obesity.

    Numerical Estimates are Skewed with Outliers:
        The mean_estimate, lower_bound, upper_bound, and ci_width columns are all right-skewed(positively skewed) and contain significant outliers at the higher end. Proper handling of these outliers and data transformation will be essential for robust statistical modeling.

    Data Quality is Generally Good (Post-Imputation): 
        After imputing the missing values in the region column, the dataset is clean with no duplicates.

    In essence, this EDA highlights age and region as the primary role on obesity patterns in this dataset, while gender appears to play a minor role. ''')

st.write('# Multivariant Analysis')
plt.figure(figsize = (25,25))
sns.pairplot(df[['mean_estimate', 'ci_width', 'year', 'lower_bound', 'upper_bound', 'age_group']], hue='age_group')
plt.suptitle('Pair Plot of Numerical Variables by Age Group', y=1.02)
st.pyplot(plt)

