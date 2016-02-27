# coding: utf-8
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

__author__ = 'fuhuamosi'

train_set = pd.read_csv('../dataset/train.csv')

# In[3]:

train_set.info()


# In[4]:

train_set.describe()


# In[5]:

fig = plt.figure(1)
fig.set_alpha(0.5)


# In[6]:

plt.subplot2grid((2, 3), (0, 0))
train_set.Survived.value_counts().plot(kind='bar')
plt.title('survived情况')
plt.ylabel('人数')


# In[7]:

plt.subplot2grid((2, 3), (0, 1))
train_set.Pclass.value_counts().plot(kind='bar')
plt.title('乘客阶级分布')
plt.ylabel('人数')


# In[8]:

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(train_set.Age, train_set.Survived)
plt.title('按年龄看survived分布')
plt.xlabel('年龄')
plt.grid(b=True, which='major', axis='x')


# In[9]:

plt.subplot2grid((2, 3), (1, 0), colspan=2)
train_set.Age[train_set.Pclass == 1].plot(kind='kde')
train_set.Age[train_set.Pclass == 2].plot(kind='kde')
train_set.Age[train_set.Pclass == 3].plot(kind='kde')
plt.xlabel("年龄")  # plots an axis label
plt.ylabel("密度")
plt.title("各等级的乘客年龄分布")
plt.legend(('头等舱', '2等舱', '3等舱'), loc='best')


# In[10]:

plt.subplot2grid((2, 3), (1, 2))
train_set.Embarked.value_counts().plot(kind='bar')
plt.title("各登船口岸上船人数")
plt.ylabel("人数")
plt.show()


# In[46]:

fig = plt.figure(2)
fig.set_alpha(0.2)
survived_0 = train_set[train_set.Survived == 0].Pclass.value_counts()
survived_1 = train_set[train_set.Survived == 1].Pclass.value_counts()
survived_df = DataFrame({'survived': survived_1, 'not survived': survived_0})
survived_df.plot(kind='bar', stacked=True)
plt.xlabel("passenger class")  # plots an axis label
plt.ylabel("people count")
plt.title("number of survived people's class")
plt.show()


# In[45]:

fig = plt.figure(3)
fig.set_alpha(0.2)
survived_0 = train_set[train_set.Survived == 0].Sex.value_counts()
survived_1 = train_set[train_set.Survived == 1].Sex.value_counts()
survived_df = DataFrame({'survived': survived_1, 'not survived': survived_0})
survived_df.plot(kind='bar', stacked=True)
plt.xlabel("passenger sex")  # plots an axis label
plt.ylabel("people count")
plt.title("number of survived people's sex")
plt.show()


# In[61]:

# 然后我们再来看看各种舱级别情况下各性别的survived情况
fig = plt.figure(4)
fig.set(alpha=0.65)  # 设置图像透明度，无所谓
plt.title("Survived on class and sex")

ax1 = fig.add_subplot(141)
train_set.Survived[train_set.Sex == 'female'][train_set.Pclass != 3].value_counts().plot(kind='bar',
                                                                                         label="female highclass",
                                                                                         color='#FA2479')
ax1.set_xticklabels(["survived", "not survived"], rotation=0)
ax1.legend(["female/high class"], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
train_set.Survived[train_set.Sex == 'female'][train_set.Pclass == 3].value_counts().plot(kind='bar',
                                                                                         label='female, low class',
                                                                                         color='pink')
ax2.set_xticklabels(["not survived", "survived"], rotation=0)
plt.legend(["female/low class"], loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)
train_set.Survived[train_set.Sex == 'male'][train_set.Pclass != 3].value_counts().plot(kind='bar',
                                                                                       label='male, high class',
                                                                                       color='lightblue')
ax3.set_xticklabels(["not survived", "survived"], rotation=0)
plt.legend(["male/high class"], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
train_set.Survived[train_set.Sex == 'male'][train_set.Pclass == 3].value_counts().plot(kind='bar',
                                                                                       label='male low class',
                                                                                       color='steelblue')
ax4.set_xticklabels(["not survived", "survived"], rotation=0)
plt.legend(["male/low class"], loc='best')

plt.show()


# In[82]:

fig = plt.figure(5)
fig.set_alpha(0.2)
survived_0 = train_set[train_set.Survived == 0].Embarked.value_counts()
survived_1 = train_set[train_set.Survived == 1].Embarked.value_counts()
survived_df = DataFrame({'survived': survived_1, 'not survived': survived_0})
survived_df.plot(kind='bar', stacked=True)
plt.xlabel("passenger embark")  # plots an axis label
plt.ylabel("people count")
plt.title("number of survived people's embark")


# In[66]:

g = train_set.groupby(by=['SibSp', 'Survived'])
df = g.count()['PassengerId']
df.plot(kind='Bar')
plt.ylabel('people count')
plt.show()


# In[95]:

survived_cabin = train_set[pd.notnull(train_set.Cabin)].Survived.value_counts()
survived_no_cabin = train_set[pd.isnull(train_set.Cabin)].Survived.value_counts()
df = DataFrame({'cabin': survived_cabin, 'not cabin': survived_no_cabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.ylabel('people count')
