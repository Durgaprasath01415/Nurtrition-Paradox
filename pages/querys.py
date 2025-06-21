import streamlit as st
import pandas as pd
import pymysql as sql

#Database connection
my_connection = sql.connect(host ="localhost", user = "root", password = "Mysql")
cur = my_connection.cursor()
cur.execute("use nutrition")

#Dictionary of questions
dict_of_question = {
    1:"1.Top 5 regions with the highest average obesity levels in the most recent year(2022)",
    2:"2.Top 5 countries with highest obesity estimates",
    3:"3.Obesity trend in India over the years(Mean_estimate)",
    4:"4.Average obesity by gender",
    5:"5.Country count by obesity level category and age group",
    6:"6.Top 5 countries least reliable countries(with highest CI_Width) ",
    6.1:"6.Top 5 countries most consistent countries (smallest average CI_Width)",
    7:"7.Average obesity by age group",
    8:"8.Top 10 Countries with consistent low obesity (low average + low CI)over the years",
    9:"9.Countries where female obesity exceeds male by large margin (same       year)",
    10:"10.Global average obesity percentage per year",
    11:"11.Avg. malnutrition by age group",
    12:"12.Top 5 countries with highest malnutrition(mean_estimate)",
    13:"13.Malnutrition trend in African region over the years",
    14:"14.Gender-based average malnutrition",
    15:"15.Malnutrition level-wise (average CI_Width by age group)",
    16:"16.Yearly malnutrition change in specific countries(India, Nigeria, Brazil)",
    17:"17.Regions with lowest malnutrition averages",
    18:"18.Countries with increasing malnutrition (💡 Hint: Use MIN() and MAX()   on Mean_Estimate per country to compare early vs. recent malnutrition levels, and filter where the difference is positive using HAVING.)",
    19:"19.Min/Max malnutrition levels year-wise comparison",
    20:"20. High CI_Width flags for monitoring(CI_width > 5)",
    21:"21.Obesity vs malnutrition comparison by country(any 5 countries)",
    22:"22.Gender-based disparity in both obesity and malnutrition",
    23:"23.Region-wise avg estimates side-by-side(Africa and America)",
    24:"24.Countries with obesity up & malnutrition down",
    25:"25.Age-wise trend analysis"}

#Dictionary of query
dict_of_query= {
    1:'''SELECT 
    REGION , AVG(MEAN_ESTIMATE) AS AVERAGE_OBESITY_LEVEL 
FROM 
    OBESITY 
WHERE 
    YEAR=2022 
GROUP BY 
    REGION 
ORDER BY AVERAGE_OBESITY_LEVEL DESC 
LIMIT 
    5;''',
    2:'''SELECT 
    COUNTRY ,MAX(MEAN_ESTIMATE) AS HIGHEST_OBESITY 
FROM 
    OBESITY 
GROUP BY 
    COUNTRY 
ORDER BY 
    HIGHEST_OBESITY DESC 
LIMIT 
    5;''',
    3:'''SELECT 
    YEAR, MEAN_ESTIMATE 
FROM 
    OBESITY 
WHERE 
    COUNTRY = 'INDIA' 
GROUP BY 
    YEAR 
ORDER BY 
    YEAR DESC;''',
    4:'''SELECT 
    GENDER,AVG(MEAN_ESTIMATE) 
FROM 
    OBESITY 
GROUP BY 
    GENDER 
ORDER BY 
    GENDER;''',
    5:'''SELECT 
    OBESITY,AGE_GROUP,COUNT(DISTINCT COUNTRY) AS COUNTRY_COUNT 
FROM 
    OBESITY 
GROUP BY 
    COUNTRY, AGE_GROUP;''',
    6:'''SELECT 
    COUNTRY ,CI_WIDTH 
FROM 
    OBESITY 
ORDER BY 
    CI_WIDTH DESC 
LIMIT 
    5;''',
    6.1:'''SELECT 
    COUNTRY , AVG(CI_WIDTH) AS AVG_CI_WIDTH 
FROM 
    OBESITY 
GROUP BY 
    COUNTRY 
ORDER BY 
    CI_WIDTH ASC 
LIMIT 
    5;''',
    7:'''SELECT 
    AGE_GROUP,AVG(MEAN_ESTIMATE) AS AVERAGE_OBESITY 
FROM 
    OBESITY 
GROUP BY 
    AGE_GROUP 
ORDER BY 
    AVERAGE_OBESITY;''',
    8:'''SELECT 
    YEAR, COUNTRY , MIN(MEAN_ESTIMATE) + MIN(CI_WIDTH) AS LOW_OBESITY 
FROM 
    OBESITY 
GROUP BY 
    YEAR, COUNTRY 
ORDER BY 
    YEAR, COUNTRY 
LIMIT 
    10;''',
    9:'''SELECT 
    YEAR,COUNTRY,AVG(MEAN_ESTIMATE) AS AVERAGE_OBESITY ,AVG(CI_WIDTH) AS AVERAGE_CI_WIDTH 
FROM 
    OBESITY 
GROUP BY 
    COUNTRY 
ORDER BY 
    AVERAGE_OBESITY, AVERAGE_CI_WIDTH  
LIMIT 
    10;''',
    10:'''SELECT 
    YEAR ,AVG(MEAN_ESTIMATE) AS AVERAGE_OBESITY 
FROM 
    OBESITY 
GROUP BY 
    YEAR 
ORDER BY 
    YEAR;''',
    11:'''SELECT 
    AGE_GROUP, MEAN_ESTIMATE 
FROM 
    MALNUTRITION 
GROUP BY 
    AGE_GROUP;''',
    12:'''SELECT 
    COUNTRY, MAX(MEAN_ESTIMATE) AS AVERAGE_MALNUTRITION 
FROM 
    MALNUTRITION 
GROUP BY 
    COUNTRY 
ORDER BY 
    AVERAGE_MALNUTRITION DESC 
LIMIT 
    10;''',
    13:'''SELECT 
    YEAR, MEAN_ESTIMATE 
FROM 
    MALNUTRITION 
WHERE 
    REGION = 'Africa' 
GROUP BY 
    YEAR 
ORDER BY 
    YEAR DESC;''',
    14:'''SELECT 
    GENDER, AVG(MEAN_ESTIMATE) 
FROM 
    MALNUTRITION 
GROUP BY 
    GENDER;''',
    15:'''SELECT 
    MALNUTRITION AS MALNUTRITION_LEVEL ,AVG(CI_WIDTH), AGE_GROUP 
FROM 
    MALNUTRITION 
GROUP BY 
    CI_WIDTH, AGE_GROUP;''',
    16:'''SELECT 
    COUNTRY,YEAR,MEAN_ESTIMATE AS MALNUTRITION_LEVEL 
FROM 
    MALNUTRITION
WHERE 
    COUNTRY IN ('India', 'Nigeria', 'Brazil') 
ORDER BY 
    COUNTRY,YEAR;''',
    17:'''SELECT 
    REGION ,AVG(MEAN_ESTIMATE) AS AVERAGE_MALNUTRITION 
FROM MALNUTRITION 
GROUP BY 
    REGION 
ORDER BY 
    AVERAGE_MALNUTRITION 
LIMIT 
    1;''',
    18:'''SELECT 
    COUNTRY, MAX(MEAN_ESTIMATE), MIN(MEAN_ESTIMATE), (MAX(MEAN_ESTIMATE)-MIN(MEAN_ESTIMATE)) AS DIFFRENCE 
FROM 
    MALNUTRITION 
GROUP BY 
    COUNTRY 
ORDER BY 
    DIFFRENCE DESC;''',
    19:'''SELECT 
    YEAR,MAX(MEAN_ESTIMATE) AS MAX, MIN(MEAN_ESTIMATE) AS MIN 
FROM 
    MALNUTRITION 
GROUP BY 
    YEAR 
ORDER BY 
    MIN;''',
    20:'''SELECT 
    COUNTRY 
FROM 
    MALNUTRITION 
WHERE 
    CI_WIDTH > 5;''',
    21:'''SELECT 
    OB.COUNTRY, AVG(OB.MEAN_ESTIMATE) AS AVERAGE_OBESITY_LEVEL 
FROM 
    OBESITY OB 
JOIN 
    MALNUTRITION MAL ON MAL.COUNTRY = OB.COUNTRY 
WHERE 
    OB.COUNTRY IN ('INDIA','UNITED KINGDOM','PERU','ITALY','ARGENTINA') 
GROUP BY 
    COUNTRY 
ORDER BY 
    COUNTRY;''',
22:'''SELECT 
    OB.GENDER,AVG(OB.MEAN_ESTIMATE) AS AVG_OBESITY,AVG(MAL.MEAN_ESTIMATE) AS AVG_MALNUTRITION 
FROM 
    OBESITY OB 
JOIN 
    MALNUTRITION MAL ON OB.GENDER = MAL.GENDER 
GROUP BY 
    OB.GENDER 
ORDER BY 
    OB.GENDER;
''',
23:'''SELECT 
    OB.COUNTRY, AVG(OB.MEAN_ESTIMATE) 
FROM 
    OBESITY OB 
JOIN 
    MALNUTRITION MAL ON OB.REGION = MAL.REGION 
WHERE 
    OB.REGION IN ('AFRICA','AMERICAS') 
GROUP BY 
    OB.REGION 
ORDER BY 
    OB.REGION;
''',
24:'''SELECT 
    OB.COUNTRY, AVG(OB.MEAN_ESTIMATE) AS AVG_OBESITY,AVG(MAL.MEAN_ESTIMATE) AS AVG_MALNUTRITION 
FROM 
    OBESITY OB 
JOIN 
    MALNUTRITION MAL ON OB.COUNTRY = MAL.COUNTRY 
GROUP BY 
    OB.COUNTRY 
ORDER BY 
    AVG_OBESITY ASC , AVG_MALNUTRITION DESC;
''',
25:'''SELECT 
    OB.AGE_GROUP, COUNT(OB.AGE_GROUP), MAL.AGE_GROUP, COUNT(MAL.AGE_GROUP) 
FROM 
    OBESITY OB 
JOIN 
    MALNUTRITION MAL ON OB.AGE_GROUP= MAL.AGE_GROUP 
GROUP BY 
    OB.AGE_GROUP;
'''}

#Function for matching the dictionary of question with dictionary of query
def matchquery(option):
    for ques_key,ques_value in dict_of_question.items():
        if option == ques_value:
            if dict_of_query.get(ques_key) != None:
                return dict_of_query.get(ques_key)
            else:
                st.warning('This query is not working', icon="⚠️")

#Function for getting data from the mysql as dataframe
def getData(question):
    query = matchquery(question)
    if st.button("Run Query",use_container_width=True,icon=":material/list_alt:" ):
        data = pd.read_sql(query,con = my_connection)
        st.dataframe(data, use_container_width=True)
        st.markdown(f"Records : {len(data)}")

#Function for showing selection box
def select():
    option = st.selectbox(
    "What query would you like to be select?",
    (dict_of_question.get(1), dict_of_question.get(2), dict_of_question.get(3), dict_of_question.get(4), dict_of_question.get(5), dict_of_question.get(6), dict_of_question.get(6.1), dict_of_question.get(7), dict_of_question.get(8), dict_of_question.get(9), dict_of_question.get(10), dict_of_question.get(11), dict_of_question.get(12), dict_of_question.get(13), dict_of_question.get(14), dict_of_question.get(15), dict_of_question.get(16), dict_of_question.get(17), dict_of_question.get(18), dict_of_question.get(19), dict_of_question.get(20),dict_of_question.get(21),dict_of_question.get(22),dict_of_question.get(23),
dict_of_question.get(24),dict_of_question.get(25)),
    index=None,
    placeholder="Select your query here...",
    )
    st.write("You selected:", option)
    getData(option)

select()


