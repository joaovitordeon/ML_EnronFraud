
# coding: utf-8

# In[386]:


import sys
import pickle
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import seaborn as sns
import numpy as np
import math
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


###Exploração sobre dados financeiros
financial_data_list=['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options','other', 'long_term_incentive', 'restricted_stock', 'director_fees']

some_financial_list=['poi','salary','bonus','total_payments','long_term_incentive','loan_advances']

data_plot = featureFormat(data_dict, some_financial_list , sort_keys = True)

_, features_df = targetFeatureSplit(data_plot)

df =pd.DataFrame(features_df, columns=['salary','bonus','total_payments','long_term_incentive','loan_advances'])


### função para plotar algumas features financeiras em forma de box plot---------------------------
def boxplots(df, ft_list):
	fig, ax = plt.subplots(1, len(ft_list), figsize=(20,6))
	for i, item in enumerate(ft_list):
		ax[i].boxplot(df[item],0, 'gD')
		ax[i].set_title(item, x=0.5 ,y=0.5, color= 'blue', rotation=0)
		ax[i].spines['right'].set_visible(False)
		ax[i].spines['top'].set_visible(False)

	plt.tick_params(bottom="off", top="off", left="off", right="off")
	plt.show()


boxplots(df, some_financial_list[1:])

## função para plotar metricas dos classifiers
def plot_metrics(df, title, legend):
    ax=df.plot(kind='bar')
    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend().set_visible(legend)
    plt.show()
#--------------------------------------------------------------------------

### Tratamento de Outliers
outliers=[]

for d in data_dict:
    if data_dict[d]['salary'] != 'NaN' and data_dict[d]['salary'] >2.5e7:  
        outliers.append(d)       

for d in data_dict:
    if data_dict[d]['bonus'] != 'NaN' and data_dict[d]['bonus'] >0.8e8:  
        outliers.append(d)

for d in data_dict:
    if data_dict[d]['total_payments'] != 'NaN' and data_dict[d]['total_payments'] >1e8:  
        outliers.append(d)

for d in data_dict:
    if data_dict[d]['long_term_incentive'] != 'NaN' and data_dict[d]['long_term_incentive'] >4e7:  
        outliers.append(d)

for d in data_dict:
    if data_dict[d]['loan_advances'] != 'NaN' and data_dict[d]['loan_advances'] >0.6e8:  
        outliers.append(d)


set(outliers)
print("-----------------------------------------------------------------------")

#data_dict["LAY KENNETH L"]["poi"] , data_dict["TOTAL"]["poi"]


## "TOTAL" é um erro de inputação de dados, e 'LAY KENNETH L' é um outlier que aparece em muitas variáveis
## Porém, ela é uma POI então não iremos excluí-la do dataset 
## "THE TRAVEL AGENCY IN THE PARK" tambem não é uma pessoa

len(data_dict)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#lista com todos os atributos menos 'email_adress'(string)
all_features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances',\
						'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',\
							 'expenses', 'exercised_stock_options','other', 'long_term_incentive',\
							  	'restricted_stock', 'director_fees','to_messages','from_poi_to_this_person',\
							  			 'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi']

### função para contar a quantidade de POI's   
def count_poi(dataset):
	poi=0
	non_poi=0
	for k in dataset.keys():
		if dataset[k]['poi']==1:
			poi+=1
		else:
			non_poi+=1	

	print("\nCount of POIs: ", poi)
	print("Count of non-POIs: ", non_poi)
	print("\n-----------------------------------------------------------------------")		

count_poi(data_dict)

###função que mostra a qualidade das features
def data_quality(dataset, f_list):
	feats=[] 
	for v in dataset.values():
		feats.append(v)
	df = pd.DataFrame(feats, columns=f_list)	
	df.replace("NaN", np.nan , inplace=True)

	count_col=[]
	for col in df.columns:
		count_col.append(df[col].count())
	
	count_col = np.array(count_col)

	locs = np.arange(len(df.columns))

	fig, ax = plt.subplots(figsize=(8,6))
	fig.subplots_adjust(bottom=0.4)

	ax.bar(locs , count_col, 0.8,  color='g')

	plt.ylabel('# of non-null observations')
	plt.title('Quality features')
	plt.xticks(locs, df.columns, rotation=90)
	plt.yticks(np.arange(0, count_col.max()+1 , 10))
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.legend().set_visible(False)
	plt.show()

	print("\n-----------------------------------------------------------------------")		

data_quality(data_dict,all_features_list)

### função que computa o melhor k (numero de features)---------------------------------------
def stats(dataframe):
    fig, ax = plt.subplots( 4, 4 , figsize=(16,7))
    fig.tight_layout()
    count=0
    for i in range(4):
        for j in range(4):
            if count < len(dataframe.columns):
                ax[i][j].hist(dataframe[dataframe.columns[count]], color='pink')
                ax[i][j].set_title(dataframe.columns[count], x=0.5 ,y=0.5, color= 'blue', rotation=0)
                ax[i][j].spines['right'].set_visible(False)
                ax[i][j].spines['top'].set_visible(False)
                ax[i][j].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                count+=1
            else:
                break

    plt.show()
    ##print(dataframe.loan_advances.value_counts())


### função para retornar metricas dos classifiers---------------------------------------------
def metrics(test_classifier,dataset, best_features, clf1, clf2, clf3):
    dic={"GaussianNB":test_classifier(clf1, dataset,best_features) , \
            "DecisionTreeClassifier":test_classifier(clf2, dataset,best_features) ,\
                 "RandomForestClassifier":test_classifier(clf3, dataset,best_features)   }
       
    df=pd.DataFrame(index=["accuracy","precision","recall"], data = dic)
    return df


### funçao que retorna uma lista com as k melhores features em ordem decrescente--------------
def best_features( fn , features, labels):
    clf = SelectKBest(k = fn)
    clf.fit(features,labels)
    
    feature_scores = {}

    for i,feat in enumerate(clf.scores_):

        feature_scores[all_features_list[1:][i]]=feat

    bf=sorted(feature_scores.items(), key=lambda x: x[1])[::-1]

    kbest=[ tup[0] for tup in bf ]

    return  kbest[:fn]

##função para deletar outliers
def delete_outliers(dataset):
    for out in [ "THE TRAVEL AGENCY IN THE PARK","TOTAL",'LOCKHART EUGENE E']:
        dataset.pop(out, None)


### função para adicionar novas features ao dataframe
def add_features(dataset):
    #porcentagem de email recebidos de POI's e enviados a POI's 
    for name in dataset:
        if dataset[name]["poi"]==1:
            dataset[name]["p_from_poi_email"] = 'NaN'
            dataset[name]["p_to_poi_email"] = 'NaN'

        else:
            if dataset[name]["from_poi_to_this_person"] != 'NaN' and  dataset[name]["to_messages"] != 'NaN': 
                dataset[name]["p_from_poi_email"] = float(dataset[name]["from_poi_to_this_person"] / dataset[name]["to_messages"])
            else:
                dataset[name]["p_from_poi_email"] = 0

            if dataset[name]["from_this_person_to_poi"] != 'NaN' and  dataset[name]["from_messages"] != 'NaN':
                dataset[name]["p_to_poi_email"] = float(dataset[name]["from_this_person_to_poi"] / dataset[name]["from_messages"])
            else: 
                dataset[name]["p_to_poi_email"] = 0

### função que adiciona as novas features a lista all_features
def add_features_to_all(features):
    features.append("p_from_poi_email")
    features.append("p_to_poi_email")

### Task 2: Remove outliers

add_features(data_dict)
delete_outliers(data_dict)
add_features_to_all(all_features_list)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
### Primeira vez para ter labels e features para usar a função best_features() 
data = featureFormat(my_dataset, all_features_list, sort_keys = True)
l, f = targetFeatureSplit(data)

##função para descobrir o melhor K
the_best_k=0
def best_k_features(f,l):
	best_f= best_features(21, f, l)
	scores={}
	for i in range(1,22):
		bf_list=["poi"]+best_f[:i]
		data = featureFormat(my_dataset , bf_list , sort_keys = False )
		labels, features = targetFeatureSplit(data)
		
		features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)
		'''cv=StratifiedShuffleSplit(n_splits=200, random_state=42)
								for train_idx, test_idx in cv.split(features,labels):
									features_train=[]
									features_test=[]
									labels_train=[]
									labels_test=[]
									for ii in train_idx:
										features_train.append(features[ii])
										labels_train.append(labels[ii])
									for jj in test_idx:
										features_test.append(features[jj])
										labels_test.append(labels[jj])'''
			
		clf=DecisionTreeClassifier()
		clf.fit(features_train,labels_train)

		scores[i]=clf.score(features_test, labels_test)

	#df = pd.DataFrame([k.values() for k in scores ], index=[k.keys() for k in scores], columns='score')
	df=pd.DataFrame.from_dict(scores, orient='index')
	global the_best_k
	the_best_k = int(df.idxmax().values)
	print("Melhor K: ",the_best_k)
	print("-----------------------------------------------------------------------\n")		
	
	plot_metrics(df,"Scores vs #Features", False)

best_k_features(f,l)
###Segunda vez para usar com as novas k-best features , adicionando o POI, pois foi retirado na funçao best_features
bf_with_poi=["poi"]+best_features(the_best_k,f,l)

data = featureFormat(my_dataset, bf_with_poi , sort_keys = True)
labels, features = targetFeatureSplit(data)


stats(pd.DataFrame(features, columns=bf_with_poi[1:]))

# [ name for name , item in data_dict.items() if item['poi']==1 ] #POI's


### Percebemos aqui que a maioria das variáveis possuem muitos valores zeros
## Isso pode ser útil para identificar outliers(possiveis POI's), porém podemos ter um problema de Data Leakage
# Nesse caso, creio que podemos usar esses zeros como uma vantagem para identificar os não POI's


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
### alguns parametros de classificadores
gnb_params={"classifier": GaussianNB(), "params": {"var_smoothing": [1e-1,1e-5,1e-9,1e-13]}}

dtc_params={"classifier": DecisionTreeClassifier(), "params": {"criterion": ["gini", "entropy"],                                                                "min_samples_split": [5,10,15,20,25]} }

rfc_params={"classifier": RandomForestClassifier() , "params": {"criterion": ["gini", "entropy"],                                                                "min_samples_split": [5,10,15,20,25]} }

all_params=[gnb_params,dtc_params,rfc_params]

best_params_list=[]

for clfr in all_params:
    try:
        clf = GridSearchCV(clfr["classifier"], clfr["params"], cv=25)
        clf.fit(features,labels)
        best_params_list.append(clf.best_params_)
       
    except (DeprecationWarning,FutureWarning) as e:
        pass

best_params_dict={"GaussianNB":best_params_list[0] ,  \
                    "DecisionTreeClassifier": best_params_list[1],  \
                        "RandomForestClassifier": best_params_list[2]  }


print("GaussianNB: ", best_params_list[0])
print("DecisionTreeClassifier: ", best_params_list[1])
print("RandomForestClassifier: ", best_params_list[2])
print("----------------------------------------------------------------------- \n")

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

## testar sem tunning
def without_tunning():
	gnb1=GaussianNB()
	dtc1=DecisionTreeClassifier()
	rfc1=RandomForestClassifier()

	df1=metrics(test_classifier, my_dataset, bf_with_poi , gnb1, dtc1, rfc1)
	print("-----------------------------------------------------------------------")

	df1=df1.transpose()

	print(df1)
	print("-----------------------------------------------------------------------")

	plot_metrics(df1,"Best metrics without tunning", True)

without_tunning()
## testar ja com os novos parametros tunados

def with_tunning():
	gnb2=GaussianNB(var_smoothing=best_params_dict["GaussianNB"]["var_smoothing"])
	dtc2=DecisionTreeClassifier(criterion=best_params_dict["DecisionTreeClassifier"]["criterion"], min_samples_split=best_params_dict["DecisionTreeClassifier"]["min_samples_split"])
	rfc2=RandomForestClassifier(criterion=best_params_dict["RandomForestClassifier"]["criterion"], min_samples_split=best_params_dict["RandomForestClassifier"]["min_samples_split"])

	df2=metrics(test_classifier, my_dataset, bf_with_poi , gnb2, dtc2, rfc2)
	print("-----------------------------------------------------------------------")

	df2=df2.transpose()

	print(df2)
	print("-----------------------------------------------------------------------")


	plot_metrics(df2,"Best metrics with tunning", True)

with_tunning()
### Aqui temos dois modelos que conseguiram um precision e um recall > .3 (RandomForest e DecisionTree)
## Porem a arvore de decisão se saiu melhor em quase todas as métricas
# Logo, escolheremos ela como nosso melhor classifier

clf = DecisionTreeClassifier(criterion=best_params_dict["DecisionTreeClassifier"]["criterion"], \
                                min_samples_split=best_params_dict["DecisionTreeClassifier"]["min_samples_split"])


# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results. 


dump_classifier_and_data(clf, my_dataset, bf_with_poi)


### Nesse caso, que tinhamos poucos dados para testar nossos classifiers, tivemos que usar métodos estatisticos
## Esses métodos ( como ShuffleSplits ) que tiram amostragens aleatórias  K vezes
# Isso permite temos maior variação no conjunto de testes e diminui a chance de enviezar nossos treinos(low bias)

