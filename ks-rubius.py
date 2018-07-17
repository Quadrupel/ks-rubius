
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

train_data = pd.read_csv('ks_train.csv')
test_data = pd.read_csv('ks_test.csv')


# In[2]:


def set_country(data):
    #Выбираем данные без ошибок
    sub_data=data.loc[data['country'] != 'N,0"']
    
    #Выбираем данные с ошибками и одной используемой валютой
    usd__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'USD']
    gbp__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'GBP']
    eur__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'EUR']
    
    cad__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'CAD']
    aud__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'AUD']
    sek__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'SEK']
    
    mxn__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'MXN']
    nzd__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'NZD']
    dkk__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'DKK']
    
    chf__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'CHF']
    nok__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'NOK']
    hkd__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'HKD']
    
    sgd__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'SGD']
    jpy__data=data.loc[data['country'] == 'N,0"'].loc[data['currency'] == 'JPY']
    
    #Изменяем страну в зависимости от валюты
    usd__data['country'].replace(to_replace='N,0"', value='US', inplace=True)
    gbp__data['country'].replace(to_replace='N,0"', value='GB', inplace=True)
    eur__data['country'].replace(to_replace='N,0"', value='DE', inplace=True)
    
    cad__data['country'].replace(to_replace='N,0"', value='CA', inplace=True)
    aud__data['country'].replace(to_replace='N,0"', value='AU', inplace=True)
    sek__data['country'].replace(to_replace='N,0"', value='SE', inplace=True)
    
    mxn__data['country'].replace(to_replace='N,0"', value='MX', inplace=True)
    nzd__data['country'].replace(to_replace='N,0"', value='NZ', inplace=True)
    dkk__data['country'].replace(to_replace='N,0"', value='DK', inplace=True)
        
    chf__data['country'].replace(to_replace='N,0"', value='CH', inplace=True)
    nok__data['country'].replace(to_replace='N,0"', value='NO', inplace=True)
    hkd__data['country'].replace(to_replace='N,0"', value='HK', inplace=True)
    
    sgd__data['country'].replace(to_replace='N,0"', value='SG', inplace=True)
    jpy__data['country'].replace(to_replace='N,0"', value='JP', inplace=True)
    
    #Объединяем датафреймы и возвращаем их
    sub_data = sub_data.append(usd__data, ignore_index=True)
    sub_data = sub_data.append(gbp__data, ignore_index=True)
    sub_data = sub_data.append(eur__data, ignore_index=True)
    
    sub_data = sub_data.append(cad__data, ignore_index=True)
    sub_data = sub_data.append(aud__data, ignore_index=True)
    sub_data = sub_data.append(sek__data, ignore_index=True)
    
    sub_data = sub_data.append(mxn__data, ignore_index=True)
    sub_data = sub_data.append(nzd__data, ignore_index=True)
    sub_data = sub_data.append(dkk__data, ignore_index=True)
    
    sub_data = sub_data.append(chf__data, ignore_index=True)
    sub_data = sub_data.append(nok__data, ignore_index=True)
    sub_data = sub_data.append(hkd__data, ignore_index=True)
    
    sub_data = sub_data.append(sgd__data, ignore_index=True)
    sub_data = sub_data.append(jpy__data, ignore_index=True)
    
    data_with_backers = sub_data.loc[data['backers'] != 0]
    data_without_money = sub_data.loc[data['backers'] == 0].loc[data['usd_pledged_real'] == 0]
    data_without_backers = sub_data.loc[data['backers'] == 0].loc[data['usd_pledged_real'] != 0]
    X = data['usd_pledged_real']//(data['usd_pledged_real'].sum()/data['backers'].sum())
    
    
    data_without_backers['backers'].replace(to_replace=0, value=X, inplace=True)
    data_without_backers['backers'].replace(to_replace=0, value=1, inplace=True)
    
    data_with_backers = data_with_backers.append(data_without_money, ignore_index=True)
    data_with_backers = data_with_backers.append(data_without_backers, ignore_index=True)
    
    return data_with_backers


# In[3]:


new_train_data = set_country(train_data)
new_test_data = set_country(test_data)


# In[4]:


from datetime import datetime
new_train_data['period'] = new_train_data['deadline'].astype('datetime64[D]') - new_train_data['launched'].astype('datetime64[D]')
new_test_data['period'] = new_test_data['deadline'].astype('datetime64[D]') - new_test_data['launched'].astype('datetime64[D]')


# In[5]:


new_train_data=new_train_data.drop('name', axis='columns')
new_train_data=new_train_data.drop('ID', axis='columns')
new_train_data=new_train_data.drop('currency', axis='columns')
new_train_data=new_train_data.drop('deadline', axis='columns')
new_train_data=new_train_data.drop('goal', axis='columns')
new_train_data=new_train_data.drop('launched', axis='columns')
new_train_data=new_train_data.drop('pledged', axis='columns')
#new_train_data=new_train_data.drop('backers', axis='columns')
new_train_data=new_train_data.drop('usd_pledged_real', axis='columns')
new_train_data=new_train_data.drop('main_category', axis='columns')

new_test_data=new_test_data.drop('name', axis='columns')
new_test_data=new_test_data.drop('ID', axis='columns')
new_test_data=new_test_data.drop('currency', axis='columns')
new_test_data=new_test_data.drop('deadline', axis='columns')
new_test_data=new_test_data.drop('goal', axis='columns')
new_test_data=new_test_data.drop('launched', axis='columns')
new_test_data=new_test_data.drop('pledged', axis='columns')
#new_test_data=new_test_data.drop('backers', axis='columns')
new_test_data=new_test_data.drop('usd_pledged_real', axis='columns')
new_test_data=new_test_data.drop('main_category', axis='columns')
new_train_data.head()


# In[6]:


from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
dicts = {}
label.fit(new_train_data['category'].drop_duplicates())
dicts['category'] = list(label.classes_)
new_train_data['category'] = label.transform(new_train_data['category'])
new_test_data['category'] = label.transform(new_test_data['category'])
label.fit(new_train_data['country'].drop_duplicates())
dicts['country'] = list(label.classes_)
new_train_data['country'] = label.transform(new_train_data['country'])
new_test_data['country'] = label.transform(new_test_data['country'])
label.fit(new_train_data['state'].drop_duplicates())
dicts['state'] = list(label.classes_)
new_train_data['state'] = label.transform(new_train_data['state'])
new_test_data['state'] = label.transform(new_test_data['state'])
label.fit(new_train_data['period'].drop_duplicates())
dicts['period'] = list(label.classes_)
new_train_data['period'] = label.transform(new_train_data['period'])
new_test_data['period'] = label.transform(new_test_data['period'])


# In[7]:


new_test_data.head()


# In[8]:


X_train = new_train_data.drop('state', axis='columns').values
y_train = new_train_data['state'].values
X_test = new_test_data.drop('state', axis='columns').values
y_test = new_test_data['state'].values


# In[9]:


from sklearn import tree

dt = tree.DecisionTreeClassifier(criterion='entropy',
                                 max_depth=100,
                                 random_state=228)
dt = dt.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=None,
                            n_estimators=275,
                            bootstrap=True,
                            random_state=228, 
                            n_jobs=-1)
rf = rf.fit(X_train, y_train)


# In[10]:


from sklearn.metrics import accuracy_score, precision_score, recall_score

test_accuracy = accuracy_score(y_test, dt.predict(X_test))
test_precision = precision_score(y_test, dt.predict(X_test))
test_recall = recall_score(y_test, dt.predict(X_test))

print('Metrics on test set:\nAccuracy = {0:.5f}\nPrecision = {1:.5f}\nRecall = {2:.5f}'
      .format(test_accuracy, test_precision, test_recall))


# In[11]:


test_accuracy = accuracy_score(y_test, rf.predict(X_test))
test_precision = precision_score(y_test, rf.predict(X_test))
test_recall = recall_score(y_test, rf.predict(X_test))

print('Metrics on test set:\nAccuracy = {0:.5f}\nPrecision = {1:.5f}\nRecall = {2:.5f}'
      .format(test_accuracy, test_precision, test_recall))


# In[12]:


from sklearn.ensemble import ExtraTreesClassifier
    
clf = ExtraTreesClassifier(n_estimators=275,
                           max_depth=None,
                           bootstrap=True, 
                           random_state=228, 
                           n_jobs=-1)
clf = clf.fit(X_train, y_train)

test_accuracy = accuracy_score(y_test, clf.predict(X_test))
test_precision = precision_score(y_test, clf.predict(X_test))
test_recall = recall_score(y_test, clf.predict(X_test))

print('Metrics on test set:\nAccuracy = {0:.5f}\nPrecision = {1:.5f}\nRecall = {2:.5f}'
      .format(test_accuracy, test_precision, test_recall))


# In[13]:


from sklearn.ensemble import GradientBoostingClassifier
    
gbc = GradientBoostingClassifier(n_estimators=275,
                                 learning_rate=1.0,
                                 max_depth=3,
                                 random_state=228)
gbc.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score

test_accuracy = accuracy_score(y_test, gbc.predict(X_test))
test_precision = precision_score(y_test, gbc.predict(X_test))
test_recall = recall_score(y_test, gbc.predict(X_test))

print('Metrics on test set:\nAccuracy = {0:.5f}\nPrecision = {1:.5f}\nRecall = {2:.5f}'
      .format(test_accuracy, test_precision, test_recall))


# In[14]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

bdt = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',
                                                 max_depth=2,
                                                 random_state=228),
                         algorithm="SAMME.R",
                         n_estimators=275,
                         random_state=228)
bdt=bdt.fit(X_train, y_train)

test_accuracy = accuracy_score(y_test, bdt.predict(X_test))
test_precision = precision_score(y_test, bdt.predict(X_test))
test_recall = recall_score(y_test, bdt.predict(X_test))

print('Metrics on test set:\nAccuracy = {0:.5f}\nPrecision = {1:.5f}\nRecall = {2:.5f}'
      .format(test_accuracy, test_precision, test_recall))


# In[ ]:


# здесь должны быть графики

