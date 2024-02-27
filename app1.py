import streamlit as st
from selenium import webdriver
import time
import numpy as np
import pandas as pd

# Load data
df1 = pd.read_csv('drugname1.csv')
df1 = df1[['Condition', 'Drug Name']]

df2 = pd.read_csv('drugsentiment.csv')
df2 = df2[['drugName', 'average_sentiment']]

df3 = pd.read_csv('drugrating.csv')
df3 = df3[['DrugName', 'Average Rating']]

def finddrug(x):
    w = df1.loc[df1['Condition'] == x]['Drug Name']
    v = w.item()
    u = v.split(',')
    return u

def sortdrug(d):
    snt = pd.DataFrame(columns=['drugName', 'average_sentiment'])
    for i in range(len(d)):
        s = df2[df2['drugName'] == d[i]]
        snt = pd.concat([snt, s])
    m = snt.sort_values(by='average_sentiment', ascending=False).reset_index(drop=True)['drugName'].tolist()
    return m

def ratedrug(m):
    r_ls = []
    for i in m:
        q = df3.loc[df3['DrugName'] == i]['Average Rating']
        r_ls.append((i, q.item()))
    df_r = pd.DataFrame(r_ls, columns=["Drug Name", "Average User Rating"])
    return df_r

def scrapdrug(w):
    with webdriver.Chrome() as driver:
        driver.get("https://www.1mg.com/search/all?name="+w)
        time.sleep(2)  # Add a delay to ensure the page is fully loaded
        try:
            element = driver.find_element_by_xpath('//a[contains(@class, "style__horizontal-card___1Zwmt")]')
            link1 = element.get_attribute("href")
            print("Link to drug details:", link1)

            driver.get(link1)

            generic_element = driver.find_element_by_xpath('//div[@class="saltInfo DrugHeader__meta-value___vqYM0"]')
            generic = generic_element.text
            print("The Generic Name of the given drug is:", generic)

            driver.get("https://www.1mg.com/search/all?name="+generic+"&filter=true&sort=rating")

            drugs = driver.find_elements_by_class_name('style__pro-title___3zxNC')
            time.sleep(15)
            drugs_list = [d.text for d in drugs]
            
            # Add print statements to debug
            print("Alternative Brand Names of Drugs:", drugs_list)

            return drugs_list
        except Exception as e:
            print(f"Error: {e}")
            return []

# ... rest of the code remains the same



# ... rest of the code remains the same

# ... (existing code)

# Streamlit App
st.title('Drug Information and Recommendation System')

# User input
option = st.radio('Choose an option:', ['Search by Condition', 'Search by Drug Name'])

# Move the instantiation of WebDriver outside the button block
driver = webdriver.Chrome()

if option == 'Search by Condition':
    condition = st.text_input('Enter a Condition')
    condition = condition.lower()

    if st.button('Generate Recommendations'):
        drugname = finddrug(condition)
        druglist = sortdrug(drugname)
        drugdata = ratedrug(druglist)

        # Display results
        st.subheader(f'Drug Names and Average User Rating for {condition}')
        st.table(drugdata)

        st.subheader('Drug Names sorted on the basis of sentiment of reviews given by users')
        st.write(druglist)

elif option == 'Search by Drug Name':
    w = st.text_input('Enter a Drug Name')

    if st.button('Get Drug Information'):
        drugs_list = scrapdrug(w)

        # Display results
        st.subheader('Alternative Brand Names of Drugs of the above generic name')
        st.write(drugs_list)

        st.subheader('Drug names are sorted according to Average Customer Rating')

# Close the WebDriver after displaying the results
driver.close()
