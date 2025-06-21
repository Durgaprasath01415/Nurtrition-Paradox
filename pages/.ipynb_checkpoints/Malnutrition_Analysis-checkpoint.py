import streamlit as st
import matplotlib.pyplot as plt
import pymysql as sql
import pandas as pd
import numpy as np
import seaborn as sns

#Loading Data
myconnection = sql.connect(host ="localhost",user = "root",password = "Mysql")
cur = myconnection.cursor()
cur.execute("use nutrition")
df = pd.read_sql("select * from nutrition.malnutrition",con = myconnection)

#Differentiating the feature into numeric and categorical
numeric_features = ['lower_bound', 'upper_bound', 'mean_estimate', 'ci_width', 'year']
categorical_features = ['region', 'gender', 'age_group', 'malnutrition']

st.write('''# Univariate Analysis Histogram Plot For Numerical Features''')
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
st.write('''# Obervation
    1) Median line of lower_bound, upper_bound, mean_estimate and ci_width are towards the left side of the plot.
    2) We can confirm from the boxplot that lower_bound, upper_bound, mean_estimate and ci_width are positive skewed(right skewed).
    3) Year has symmetric distribution and Median line is almost in mid of the distribution plot.
    4) lower_bound, upper_bound, mean_estimate and ci_width columns have many outlier data point ''')

#Function to get the number of outlier in the columns
def count_outliers_iqr(data):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return len(outliers)

st.write('# Number of Outlier')
#Seeing the number of outlier in columns
st.write(f'lower_bound : {count_outliers_iqr(df['lower_bound'])}')
st.write(f'upper_bound : {count_outliers_iqr(df['upper_bound'])}')
st.write(f'mean_estimate : {count_outliers_iqr(df['mean_estimate'])}')
st.write(f'ci_width : {count_outliers_iqr(df['ci_width'])}')

st.write('# Univariate Analysis Count Plot Categorical Feature')
# Histo plot for Categorical columns
plt.figure(figsize=(18, 25))
for i, col in enumerate(categorical_features):
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
    2) 'Western Pacific', 'Eastern Mediterranean' have less number of entries then 'Europe', 'Africa', and 'Americas'.
    3) 'South-East Asia' and 'Unknown' category (imputed values) seems similar number of entries.

## gender:

    1) The counts for 'Male', 'Female', and 'Both' genders are almost perfectly equal. 
    2) This indicates a balanced representation of gender categories in the dataset.

## age_group:

    1) The 'Child' age group has a higher count compared to the 'Adult' age group.
    2) Since child count is higher, the data of childer's malnutrition data is more then the adult's in dataset.

## malnutrition:

    1) 'Low' malnutrition category is higher, compare to 'High' and 'Moderate' categories combined. Which shows in dataset of 'low' malnutrition entries are higher.
    2) 'Moderate' malnutrition has a higher count than 'High' malnutrition.
    3) 'High' malnutrition has a leastest count.''')

st.write('# Bivariate Analysis HeatMap Between Numericals Feature')
# 1. Numerical vs. Numerical
# Correlation Matrix and Heatmap
numerical_cols = ['lower_bound', 'upper_bound', 'mean_estimate', 'ci_width', 'year']
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.xticks(rotation=45, ha='right')
plt.title('Correlation Matrix of Numerical Variables')
st.pyplot(fig)

st.write('''# Observations from the correlation matrix heatmap of the numerical variables:

## Strong Positive Correlations (Near 1.0):

    mean_estimate is highly correlated with both lower_bound (0.94) and upper_bound (0.94). This is expected, as mean_estimate is the central value between lower_bound and upper_bound, and these three variables describe the same estimated range.
    ci_width and upper_bound also show a strong positive correlation with each other (0.81).
    lower_bound and upper_bound have moderate positive correlation with each other (0.77)

## Moderate Positive Correlation (Around mid 0.5):

    ci_width (confidence interval width) shows a moderate positive correlation with mean_estimate (0.55). 
    
## Weak Positive Correlations (Near 0):

    year has a weak positive correlation with ci_width (0.13), upper_bound (0.04), and mean_estimate (0.11).
    
## Negative Correlations:

    lower_bound (-0.08) and mean_estimate (-0.02) shows a very weak and negative correlation with year.''')

st.write('# Bivariate Analysis Bar Plot Between Categorical and Numericals Features')
# 2. Categorical vs. Numerical
# Box plots of mean_estimate and ci_width across categorical variables
categorical_num_pairs = [
    ('region', 'mean_estimate'),
    ('gender', 'mean_estimate'),
    ('age_group', 'mean_estimate'),
    ('malnutrition', 'mean_estimate'),
    ('region', 'ci_width'),
    ('gender', 'ci_width'),
    ('age_group', 'ci_width'),
    ('malnutrition', 'ci_width')
]

plt.figure(figsize=(18, 15))
for i, (cat_col, num_col) in enumerate(categorical_num_pairs):
    plt.subplot(4, 2, i + 1)
    sns.barplot(x=cat_col, y=num_col, data=df)
    plt.title(f'{num_col} by {cat_col}')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.suptitle('Categorical vs. Numerical Variables: Box Plots', y=1.02)
st.pyplot(plt)
st.write('''# Observations from the bar plots 
## mean_estimate and ci_width across different categorical variables:

    1.mean_estimate and region:
        'South-East Asia' regions show the highest mean_estimate values, indicating high malnutrition percentages in these regions.
        'Eastern mediterranean' and 'Africa' have moderate mean_estimate values .
        'Americas', 'Western Pacific' and 'Europe' have low mean_estimate values.
        The 'Unknown' region has an mean_estimate somewhere in the middle, which is representing countries whose regions were initially missing.

    2.mean_estimate and gender:
        Mean_estimate along the gender varies little towards each gender.
        'Male' gender have the higher mean_estimate then other gender.
        'Female' gender has the least mean_estimate.
        'Both' gender inbetween the 'male' and 'female' gender.

    3.mean_estimate and age_group:
        The 'Adult' age group has a significantly higher mean_estimate compared to the 'Child' age group. This clearly indicates that malnutrition percentages much higher in adults than in children.

    4.mean_estimate and malnutrition level:
        The mean_estimate values are clearly separated by the malnutrition categories: 'High' has the highest, 'Moderate' is in the middle, and 'Low' has the lowest mean_estimate.

    5.ci_width and region:
        'South-East Asia' and 'Africa' show high level in ci_width values.
        'Eastern mediterranean' have moderately lesser level then 'South-East Asia' and 'Africa'.
        'America' and 'Western Pacific' have almost similar ci_width values.
        'Unknown' and 'Europe' have the least ci_width values.

    6.ci_width by gender:
        The ci_width values are high in "Male','Female' is slight less then 'Male' and 'Both' in slight less then 'Female'.

    7.ci_width by age_group:
        The 'Adult' age group has a noticeably higher ci_width than the 'Child' age group.

    8.ci_width by malnutrition:
        'Moderate' has the highest, 'High' is in the middle, and 'Low' has the lowest ci_width.''')

st.write('# Bivariate Analysis Bar Plot Between Categorical Features')
# 3. Categorical vs. Categorical

# Stacked Bar Charts for categorical relationships
# Region vs. malnutrition
df_region_malnutrition = pd.crosstab(df['region'], df['malnutrition'], normalize='index')
df_region_malnutrition.plot(kind='bar', stacked=True, figsize=(10, 4))
plt.title('Malnutrition Distribution by Region')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.show()

st.pyplot(plt)
st.write('''## Observations from the stacked bar 'malnutrition Distribution by Region':

    Across 'Americas', 'Africa', 'Eastern mediterranean', 'Europe', 'Unknown' and 'Western Pacific' regions, the 'Low' malnutrition category as the largest proportion.

    'South-East Asia' show a visibly larger proportion of 'Moderate' and 'High' malnutrition compared to other regions.

    'Africa' and 'Unknown' also have 'High' and 'Moderate' malnutrition, but their proportions are smaller than 'South-East Asia'.

    'Americas' and 'Western Pacific' has a very small proportion 'Moderate' malnutrition, with 'Low' almost the entire bar.

    'Europe' region show has only the 'Low' malnutrition.''')

# Gender vs. Obesity
# Gender vs. malnutrition
df_gender_malnutrition = pd.crosstab(df['gender'], df['malnutrition'], normalize='index')
df_gender_malnutrition.plot(kind='bar', stacked=True, figsize=(8, 5))
plt.title('Malnutrition Distribution by Gender')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
st.pyplot(plt)
st.write('''## Observations from the stacked bar chart of 'Malnutrition Distribution by Gender':

    For all gender categories ('Male', 'Female', and 'Both'), the 'Low' malnutrition category is majorly observed.

    'Female' and 'Both' categories have almost a similar proportion of 'high' malnutrition with 'Male' being bit more then other two gender.

    Categories of the gender shows that malnutrition is not only relating towards one or two gender categories but all gender have the 'low' malnutrition and 'Moderate' malnutrition.''')

# Age Group vs. Obesity
df_agegroup_malnutrition = pd.crosstab(df['age_group'], df['malnutrition'], normalize='index')
df_agegroup_malnutrition.plot(kind='bar', stacked=True, figsize=(8, 5))
plt.title('Malnutrition Distribution by Age Group')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
st.pyplot(plt)
st.write('''## Observations from the stacked bar chart of 'Malnutrition Distribution by Age Group':

    This plot reveals a very strong and clear relationship between age_group and the distribution of malnutrition categories.

    'Child' age_group is majorly of 'Low' malnutrition level.
    
    'Moderate' malnutrition is high in Adult then Children.
    
    The proportions of 'High' malnutrition in both Adult and Children are extremely small.''')


st.write('''# Summary of the EDA's of malnutrition dataset

The EDA revealed a dataset rich in demographic(age_group) and geographical(region) information related to malnutrition

    The majority of observations fall into the 'Low' malnutrition category. However, the 'High' and 'Moderate' categories are comparatively  smaller, represent crucial areas for intervention and study. This imbalance is a primary concern for predictive modeling.

    Age is the demographic factor:

        Adults show significantly higher malnutrition estimates and a much greater proportion of 'High' and 'Moderate' malnutrition compared to 'Children'. 
        This suggests that interventions and public health strategies related to malnutrition should strongly prioritize the adult population.

    Regional is the geographical factor:

        'South-East Asia' regions exhibit the highest malnutrition estimates and the largest proportions of 'High' and 'Moderate' malnutrition.These region likely require attention for malnutrition prevention and management.
        Europe stands out with almost 'Low' malnutrition observations.

    Gender Shows Minimal Impact on malnutrition: 
        Across the all three categories of gender, the proportional distribution of 'Male' gender categories have 'High' and 'Moderate' malnutrition more compare to 'Female', and 'Both' genders. 
        Gender 'Both' have comparitively higher 'High' and 'Moderate' then 'Female' gender.

    Numerical Estimates are Skewed with Outliers:
        The mean_estimate, lower_bound, upper_bound, and ci_width columns are all right-skewed(positively skewed) and contain significant outliers at the higher end. Proper handling of these outliers and data transformation will be essential for robust statistical modeling.

    Data Quality is Generally Good (Post-Imputation): 
        After imputing the missing values in the region column, the dataset is clean with no duplicates.

    In essence, this EDA highlights age and region as the primary role on malnutrition patterns in this dataset, while gender appears to play a minor role. ''')

st.write('# Multivariant Analysis')
sns.pairplot(df[['mean_estimate', 'ci_width', 'year', 'lower_bound', 'upper_bound', 'age_group']], hue='age_group')
plt.suptitle('Pair Plot of Numerical Variables by Age Group', y=1.02)
st.pyplot(plt)

