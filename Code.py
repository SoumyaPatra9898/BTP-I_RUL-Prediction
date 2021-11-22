#!/usr/bin/env python
# coding: utf-8

# In[2]:


import scipy.io
import numpy as np

mat = scipy.io.loadmat('dbhole.mat')

mat.items()

#mat["d1h1"]
d1h1=mat["d1h1"]


d1h2=mat["d1h2"]
a=mat.keys()
#print(a)

print(len(mat["d1h1"]))

#print(len(d1h13))
#print(mat)
#print(d1h1.shape)
'''
X = np.concatenate([d1h1, d1h2])

lengths = [len(d1h1), len(d1h2)]

print(X)

print(X.shape)
'''


# In[3]:


import pandas as pd
from hmmlearn import hmm
from hmmlearn import base
import numpy as np
#from sklearn.externals import joblib
#import matplotlib.pyplot as plt
from matplotlib import cm, pyplot as plt
import math
#import loglikelihood


# In[4]:


np.random.seed(123)

hval = [21,20,20,16,19,18,21,19,23,23,23,21]
data = []
data1=[]
for i in range(12):
    data1.append(mat['d' + str(i+1) + "h2"])
    for j in range(hval[i]):
        data.append(mat['d' + str(i+1) + 'h' + str(j + 1)])


# In[1]:



        #count += 1
        #data[i] = scale(data[i])
#data = pd.concat(data)
remodel = hmm.GaussianHMM(n_components= 3, covariance_type="full", n_iter=500).fit(data[2])
#print data[1]
#remodel.startprob_ = np.array([0.6, 0.3, 0.1]
#remodel.transmat_ = np.array([[0.7, 0.2, 0.1],[0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
#remodel.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
#remodel.covars_ = np.tile(np.identity(2), (3, 1, 1))
#data[1], Z = model.sample(100)
#print data[2]
pval = [] 
hidden_states = remodel.predict(data[36])
#p = 1
#remodel.fit(data[2])
#print remodel
##fig, axs = plt.subplots(remodel.n_components, sharex=True, sharey=True)
#plt.show()
#base._BaseHMM._init(data[1])
#print "ll = " + str(base._BaseHMM._compute_log_likelihood(data[1]))
print("Means and vars of each hidden state")
for i in range(remodel.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", remodel.means_[i])
    print("var = ", np.diag(remodel.covars_[i]))
    #print()

#fig, axs = plt.subplots(remodel.n_components, sharex=True, sharey=True)
#colours = cm.rainbow(np.linspace(0, 1, remodel.n_components))
#for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    #mask = hidden_states == i
    #ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
    #ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    #ax.xaxis.set_major_locator(YearLocator())
    #ax.xaxis.set_minor_locator(MonthLocator())

    #ax.grid(True)
'''
f = open('Hidden.txt', 'w')
for i in hidden_states:
	f.write(str(i) + " ")
f.close()
#plt.show()
print ("remodel = " + str(remodel.monitor_.converged))
#loglh =  loglikelihood.llr(np.matrix(remodel.transmat_))
#print loglh

for itr in xrange(0,21):
	#remodel = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=500).fit(data[itr])
	Z = remodel.predict(data[itr])
	#joblib.dump(remodel, "filename.pkl")
	#print Z
	#print remodel.transmat_
	print Z
	p = 1
	for i in xrange(len(Z)-1):
		p *= remodel.transmat_[Z[i]][Z[i+1]]
	pval.append(math.log(p))
	#loglh =  loglikelihood.llr(np.matrix())
	#print loglh
	print math.log(p)
	#print p
#for itr in xrange(0,len(pval)-1):
#	pval[itr] = pval[itr]+pval[itr+1]
#	pval[itr] = pval[itr]/2

pval_mean = np.mean(pval)
pval_sd = np.std(pval)
y = np.linspace(1,len(pval), len(pval))
plt.plot(y,pval)
plt.plot(pval_mean-pval_sd)
plt.show()
#print "pval_mean = " + str(pval_mean)
#print "pval_sd = " + str(pval_sd)
#threshold = -19.355
threshold = pval_mean 
th1 = -19.355
print "Th = " + str(threshold)
state = []
for i in pval:
	if i > threshold:
		state.append("EX")
	else:
		state.append("NO")
print state
f = open('states2.txt', 'ab')
#for i in state:
f.write('\n' + str(state))
f.close()
print remodel.transmat_
remodel.transmat_[Z[1]][Z[2]]'''


# In[202]:


remodel = hmm.GaussianHMM(n_components= 2, covariance_type="full", n_iter=500).fit(data[2])


# In[30]:


remodelbase=base._BaseHMM(n_components=2,n_iter=500).fit(data[2],len(data[2]))


# In[60]:


x=base._BaseHMM(n_components= 2, n_iter=500)._init(data[2],len(data[2]))


# In[64]:


print(x._compute_log_likelihood(data[2]))


# In[69]:


x.score(data[2])


# In[70]:


x=base._BaseHMM(n_components= 2, n_iter=500).fit(data[2])


# In[84]:


for i in range(hval[0]):
    print(remodel.score(data[i]))
#remodel.predict_proba(data[1])


# In[78]:


print("Means and vars of each hidden state")
for i in range(remodel.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", remodel.means_[i])
    print("var = ", np.diag(remodel.covars_[i]))


# In[98]:


pval=[]
for itr in range(0,21):
	#remodel = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=500).fit(data[itr])
	Z = remodel.predict(data[itr])
	#joblib.dump(remodel, "filename.pkl")
	#print Z
	#print remodel.transmat_
	print (Z)
	p = 1
	for i in range(len(Z)-1):
		p *= remodel.transmat_[Z[i]][Z[i+1]]
	pval.append(math.log(p))
	#loglh =  loglikelihood.llr(np.matrix())
	#print loglh
	print (math.log(p))
	#print p
#for itr in xrange(0,len(pval)-1):
#	pval[itr] = pval[itr]+pval[itr+1]
#	pval[itr] = pval[itr]/2

pval_mean = np.mean(pval)
pval_sd = np.std(pval)
y = np.linspace(1,len(pval), len(pval))
plt.plot(y,pval)
plt.plot(pval_mean-pval_sd)
plt.show()
#print "pval_mean = " + str(pval_mean)
#print "pval_sd = " + str(pval_sd)
#threshold = -19.355
threshold = pval_mean-2*pval_sd
th1 = -19.355
print ("Th = " + str(threshold))
state = []
for j in range(0,21):
    i=remodel.score(data[j])
    if i > threshold:
        state.append("EX")
    else:
        state.append("NO")
print (state)


# In[99]:


remodel.fit(data[23])


# In[100]:


for i in range(hval[0]):
    print(remodel.score(data[i]))


# In[210]:


from sklearn import preprocessing
data0=[]
for i in range(len(data)):
    scaler = preprocessing.StandardScaler().fit(data[i])
    data0.append(scaler.transform(data[i]))


# In[129]:


#print(data0[0][0])
plt.scatter(data[0][:,0],data[0][:,1])
plt.xlabel('thrust force')
plt.ylabel('torque')
#plt.show()


# In[115]:


remodel.fit(X_scaled)


# In[116]:


remodel.score(X_scaled)


# In[118]:


remodel.score(data[0])


# In[220]:


#remodel.fit(data1)
np.shape(data1)


# In[180]:


for i in range(0,len(data)):
    print(np.shape(data0[i]))
print(data0[0])


# In[221]:


remodel = hmm.GaussianHMM(n_components= 2, covariance_type="full", n_iter=500).fit(data0[0])
prob=[]
x=0
for i in range(0,12):
    prob.append(remodel.score(data0[x+1]))
    x=x+hval[i]
    print(prob[i])


# In[222]:


mean=np.mean(prob)
print(mean)


# In[223]:


sd=math.sqrt(np.var(prob))
print(sd)


# In[224]:


k=2
threshold=mean-2*sd
print(threshold)


# In[225]:


x=0
for i in range(0,12):
    #prob.append(remodel.score(data0[x+1]))
    #print(remodel.score(data0[x+1]))
    print(np.shape(data0[x+1]))
    x=x+hval[i]


# In[228]:


x=0
hmm2_idx=[]
for i in range(0,12):
    for j in range (1,hval[i]):
        if remodel.score(data0[x+j]) < threshold :
            hmm2_idx.append(j)
            print(j)
            x=x+hval[i]
            break
        


# In[199]:


arr=np.zeros((12,226,2))
arr[0]=np.array(data0[0])
print(arr)
'''

x=0
for i in range(0,12):
    prob.append(remodel.score(data0[x+1]))
    x=x+hval[i]
    '''


# In[255]:


remodel2 = hmm.GaussianHMM(n_components= 2, covariance_type="full", n_iter=500).fit(data0[hmm2_idx[0]])
prob2=[]
y=0
for i in range(0,12):
    x=hmm2_idx[i]
    prob2.append(remodel2.score(data0[x+y]))
    y=y+hval[i]
    print(prob2[i])


# In[256]:


mean2=np.mean(prob2)
sd2=math.sqrt(np.var(prob2))
print(mean2,sd2)


# In[257]:


k=2
threshold2=mean2-2*sd2
print(threshold2)


# In[250]:


x=0
for i in range(0,12):
    #prob.append(remodel.score(data0[x+1]))
    #print(remodel.score(data0[x+1]))
    y=hmm2_idx[i]
    print(np.shape(data0[x+y]))
    x=x+hval[i]


# In[263]:


x=0
hmm3_idx=[]
for i in range(0,12):
    y=hmm2_idx[i]
    for j in range (y,hval[i]):
        if j==hval[i]-1 or remodel2.score(data0[x+j]) < threshold2 :
            hmm3_idx.append(j)
            print(j)
            #if(i==2):
            #    print(remodel2.score(data0[x+j]))
            x=x+hval[i]
            break
    if j==hval[i]:
        print(j)


# In[233]:


print(remodel.score(mat['d4h3']),remodel2.score(mat['d4h3']))


# In[260]:


remodel3 = hmm.GaussianHMM(n_components= 2, covariance_type="full", n_iter=500).fit(data0[hmm3_idx[0]])
prob3=[]
y=0
for i in range(0,12):
    x=hmm3_idx[i]
    prob3.append(remodel3.score(data0[x+y]))
    y=y+hval[i]
    print(prob3[i])


# In[253]:


mean3=np.mean(prob3)
sd3=math.sqrt(np.var(prob3))
print(mean3,sd3)
k=2
threshold3=mean3-2*sd3
print(threshold2)


# In[254]:


x=0
hmm4_idx=[]
for i in range(0,12):
    y=hmm3_idx[i]
    for j in range (y,hval[i]):
        if remodel.score(data0[x+j]) < threshold3 :
            hmm4_idx.append(j)
            print(j)
            #if(i==2):
            #    print(remodel2.score(data0[x+j]))
            x=x+hval[i]
            break


# In[5]:


df=pd.read_csv('dh_features_final.csv')


# In[6]:


arr=df.to_numpy()
arr


# In[275]:


print(arr)


# In[152]:


xtrain,ytrain=arr[0:200,1:28],arr[0:200,28]


# In[153]:


print(ytrain)


# In[9]:


k=0
for i in range(10):
    k+=hval[i]
print(k)
print(hval[10]+hval[11])


# In[10]:


xtest,ytest=arr[200:244,1:28],arr[200:244,28]


# In[11]:


print(ytest)


# In[12]:


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(xtrain, ytrain)
y_pred = model.predict(xtest)


# In[13]:


print(y_pred)


# In[14]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(xtrain, ytrain)
y_pred = model.predict(xtest)


# In[15]:


print(y_pred)


# In[16]:


print(pipe.score(xtest,ytest))


# In[17]:


print(model.score(xtest,ytest))


# In[18]:


idx=np.arange(0,44)
print(idx)


# In[19]:



plt.figure(figsize=(10,10))
plt.scatter(idx,y_pred, c='crimson')
plt.scatter(idx,ytest)
#plt.yscale('log')
#plt.xscale('log')

#p1 = max(max(y_pred), max(ytest))
#p2 = min(min(y_pred), min(ytest))
#plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[20]:



from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
regr = RandomForestRegressor()
regr.fit(xtrain, ytrain)


# In[21]:


y_pred_rand=regr.predict(xtest)
print(regr.score(xtest,ytest))
print(y_pred_rand)


# In[22]:


plt.figure(figsize=(10,10))
plt.scatter(idx,y_pred_rand, c='crimson')
plt.scatter(idx,ytest)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[23]:


actual = ytest[18:23]+ytest[39:44]
predicted = y_pred_rand[18:23]+y_pred_rand[39:44]
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(actual, predicted)

rmse = math.sqrt(mse)
print(rmse)


# In[59]:


actual = ytest[18:23]+ytest[39:44]
predicted = y_pred[18:23]+y_pred[39:44]
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(actual, predicted)

rmse = math.sqrt(mse)
print(rmse)


# In[25]:


from xgboost import XGBRegressor as xgb
xg_reg = xgb(objective ='reg:squarederror')
xg_reg.fit(xtrain,ytrain)
y_pred_xgb = xg_reg.predict(xtest)


# In[337]:


pip install xgboost


# In[339]:


pip --update xgboost


# In[341]:


pip install --upgrade xgboost


# In[26]:


import xgboost as xgb


# In[27]:


print(y_pred_xgb)


# In[28]:


plt.figure(figsize=(10,10))
plt.scatter(idx,y_pred_xgb, c='crimson')
plt.scatter(idx,ytest)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[30]:



rmse = np.sqrt(mean_squared_error(ytest, y_pred_xgb))
print(rmse)


# In[40]:


plt.figure(figsize=(10,10))
plt.plot(idx[:23],y_pred_xgb[:23], c='crimson',label='predicted',marker='x')
plt.plot(idx[:23],ytest[:23],label='real',marker='.')
plt.xlabel('Driller Hole 11', fontsize=15)
plt.ylabel('RUL', fontsize=15)
plt.legend()
plt.axis('equal')
plt.show()


# In[55]:


def plotit(x,y1,y2,x_label,Title):
    plt.figure(figsize=(7,7))
    plt.plot(x,y1, c='crimson',label='predicted',marker='x')
    plt.plot(x,y2,label='real',marker='.')
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel('RUL', fontsize=15)
    plt.legend()
    plt.title(Title)
    plt.axis('equal')
    plt.show()


# In[56]:


plotit(idx[:23],y_pred[:23],ytest[:23],'Driller 11','Linear Regression')


# In[57]:


plotit(idx[23:],y_pred[23:],ytest[23:],'Driller 12','Linear Regression')


# In[60]:


plotit(idx[23:],y_pred_rand[23:],ytest[23:],'Driller 12','Random Forest')


# In[61]:


plotit(idx[:23],y_pred_rand[:23],ytest[:23],'Driller 11','Random Forest')


# In[62]:


plotit(idx[:23],y_pred_xgb[:23],ytest[:23],'Driller 11','xgboost')


# In[63]:


plotit(idx[23:],y_pred_rand[23:],ytest[23:],'Driller 12','xgboost')


# In[66]:


print(regr.score(xtrain,ytrain))
y_train=regr.predict(xtrain)
print(y_train)


# In[68]:


plotit(idx[:41],y_train[:41],ytrain[:41],'Driller 1','linear')


# In[70]:


print(np.sqrt(mean_squared_error(ytest, y_pred_xgb)))


# In[71]:


print(np.sqrt(mean_squared_error(ytrain, y_train)))


# In[76]:


yround=[]
for i in range(len(y_train)):
    yround.append(round(y_train[i]))


# In[78]:


print(np.sqrt(mean_squared_error(ytrain, yround)))


# In[81]:


from sklearn.cluster import KMeans
np.shape(arr)
#X=np.transpose(arr)


# In[89]:


X=arr[:,1:]


# In[90]:


print(X)


# In[94]:


k_means = KMeans(n_clusters=4, random_state=42, init = 'random').fit(X)
clusters = k_means.predict(X)
clusters


# In[139]:


x=0
l=[]
for i in range(12):
    t=[]
    for j in range(hval[i]):
        t.append(clusters[x+j])
    l.append(t)
    x=x+hval[i]
print(l)


# In[115]:


X_=X.tolist()
X_=X_[0:244]
for i in range(len(data)):
    X_[i].append(len(data[i]))


# In[140]:


X1=np.array(X_)
k_means1 = KMeans(n_clusters=4, random_state=42, init = 'random').fit(X1)
clusters1 = k_means1.predict(X1)
clusters1


# In[141]:


x=0
l1=[]
for i in range(12):
    t=[]
    for j in range(hval[i]):
        t.append(clusters1[x+j])
    l1.append(t)
    x=x+hval[i]
print(l1)


# In[118]:


X1=np.array(X_)
k_means1 = KMeans(n_clusters=3, random_state=42, init = 'random').fit(X1)
clusters1 = k_means1.predict(X1)
clusters1


# In[119]:


x=0
l1=[]
for i in range(12):
    t=[]
    for j in range(hval[i]):
        t.append(clusters1[x+j])
    l1.append(t)
    x=x+hval[i]
print(l1)


# In[165]:


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
gaussian_model = GaussianMixture(n_components=4)

# train the model
gaussian_model.fit(X1)

# assign each data point to a cluster
gaussian_result = gaussian_model.predict(X1)
gaussian_result


# In[166]:


x=0
l2=[]
for i in range(12):
    t=[]
    for j in range(hval[i]):
        t.append(gaussian_result[x+j])
    l2.append(t)
    x=x+hval[i]
print(l2)


# In[126]:


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import MeanShift

# initialize the data set we'll work with


# define the model
mean_model = MeanShift()

# assign each data point to a cluster
mean_result = mean_model.fit_predict(X1)

# get all of the unique clusters
mean_clusters = unique(X1)


# In[127]:


mean_clusters


# In[128]:


for mean_cluster in mean_clusters:
    # get data points that fall in this cluster
    index = where(mean_result == mean_cluster)
    # make the plot
    pyplot.scatter(X1[index, 0], X1[index, 1])

# show the Mean-Shift plot
pyplot.show()


# In[130]:


plt.scatter(data[20][:,0],data[20][:,1])
plt.xlabel('thrust force')
plt.ylabel('torque')


# In[131]:


plt.scatter(data[10][:,0],data[10][:,1])
plt.xlabel('thrust force')
plt.ylabel('torque')


# In[145]:


for i in range(len(l1)):
    for j in range(len(l1[i])):
        if j<8:
            print(l1[i][j], end="  ")
        else:
            print(l1[i][j], end="   ")
    print("\n")


# In[146]:


print(math.sqrt(mean_squared_error(ytest,y_pred)))


# In[148]:


print(math.sqrt(mean_squared_error(ytest[18:23]+ytest[39:44],y_pred[18:23]+y_pred[39:44])))


# In[154]:


y_train=model.predict(xtrain)
print(math.sqrt(mean_squared_error(ytrain,y_train)))


# In[155]:


print(regr.score(xtest,ytest))


# In[159]:


print(xg_reg.score(xtest,ytest))


# In[161]:


y_train=xg_reg.predict(xtrain)
print(math.sqrt(mean_squared_error(ytrain,y_train)))


# In[162]:


print(math.sqrt(mean_squared_error(ytest[18:23]+ytest[39:44],y_pred_xgb[18:23]+y_pred_xgb[39:44])))


# In[163]:


print(math.sqrt(mean_squared_error(ytest,y_pred_rand)))


# In[164]:


for i in range(len(l2)):
    for j in range(len(l2[i])):
        if j<8:
            print(l2[i][j], end="  ")
        else:
            print(l2[i][j], end="   ")
    print("\n")


# In[168]:


for i in range(len(l2)):
    for j in range(len(l2[i])):
        if j<8:
            print(l2[i][j], end="  ")
        else:
            print(l2[i][j], end="   ")
    print("\n")


# In[177]:


x=idx[:21]
plt.figure(figsize=(7,7))
plt.plot(x,y_pred[23:], c='crimson',label='Linear Regression',marker='x')
plt.plot(x,ytest[23:],label='Real RUL',marker='.',c='black')
plt.plot(x,y_pred_rand[23:], c='green',marker='+',label='Random Forest')
plt.plot(x,y_pred_xgb[23:],c='blue', marker='o', label='xgboost')
plt.xlabel('Hole#', fontsize=17)
plt.ylabel('RUL', fontsize=17)
plt.legend()
plt.title('Drill bit -12', fontsize=17)
plt.axis('equal')
plt.show()


# In[ ]:




