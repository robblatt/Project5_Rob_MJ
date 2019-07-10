import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.feature_selection import SelectKBest, f_regression,mutual_info_regression


def feature_engineering(data):
    
    df = data
    
    print('creating and binning columns')
    
    df['EDUCATION_Y']=df['EDUCATION'].isin([1,2]).astype(int)
    df['EDUCATION_N']=df['EDUCATION'].isin([3,4,5,6]).astype(int)

    df['AGE_to_28']=df['AGE'].isin(list(range(0,29))).astype(int)
    df['AGE_29_34']=df['AGE'].isin([29,30,31,32,33,32,33,34]).astype(int)
    df['AGE_35_42']=df['AGE'].isin([35,36,37,38,39,40,41,42]).astype(int)
    df['AGE_43_above']=df['AGE'].isin(list(range(43,100))).astype(int)

#     ----
    
#     dummies = pd.get_dummies(df['pay_score_sums'], prefix='score')
#     df_dumm = pd.concat([df, dummies], axis = 1)

#     ---
    
    
    print('transforming pay and bill columns')
    
    df.rename(columns = {'PAY_AMT1':"aug_bill_payment",
                         'PAY_AMT2': "jul_bill_payment", 
                         'PAY_AMT3': "jun_bill_payment", 
                         'PAY_AMT4': "may_bill_payment", 
                         'PAY_AMT5': "apr_bill_payment", 
                         'PAY_AMT6': "mar_bill_payment", 
                         'BILL_AMT1': "sep_bill", 
                         'BILL_AMT2': "aug_bill", 
                         'BILL_AMT3': "jul_bill", 
                         'BILL_AMT4': "jun_bill", 
                         'BILL_AMT5': "may_bill", 
                         'BILL_AMT6': "apr_bill",}, inplace = True)
    
    print('creating pay_score_sum column')
    
    col_list = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'] 
    df['pay_score_sum'] = df[col_list].sum(axis=1)
    
    
    bins = [-3, -1, 3, 10, 100]                                                                                                   
    labels = ['1', '2','3','4']
    df['pay_score_sums'] = pd.cut(df['pay_score_sum'], bins = bins, labels = labels)

    print('creating monthly balances')
    
    df['bal_1'] = df['aug_bill'] - df['aug_bill_payment']
    df['bal_2'] = df['jul_bill'] - df['jul_bill_payment']
    df['bal_3'] = df['may_bill'] - df['may_bill_payment']
    df['bal_4'] = df['apr_bill'] - df['apr_bill_payment']

    print('determining monthly credit balances over 80% of limit ')
    
    length = len(df)
    
    bal_1_ex = np.zeros(length)
    for i in range(len(df['bal_1'])):
        bal_1_ex[i] = df['bal_1'][i] > df['LIMIT_BAL'][i] * .8
    df['bal_1_ex'] = bal_1_ex

    bal_2_ex = np.zeros(length)
    for i in range(len(df['bal_2'])):
        bal_2_ex[i] = df['bal_2'][i] > df['LIMIT_BAL'][i] * .8
    df['bal_2_ex'] = bal_2_ex

    bal_3_ex = np.zeros(length)
    for i in range(len(df['bal_3'])):
        bal_3_ex[i] = df['bal_3'][i] > df['LIMIT_BAL'][i] * .8
    df['bal_3_ex'] = bal_3_ex

    bal_4_ex = np.zeros(length)
    for i in range(len(df['bal_4'])):
        bal_4_ex[i] = df['bal_4'][i] > df['LIMIT_BAL'][i] * .8
    df['bal_4_ex'] = bal_4_ex
    
    print('creating credit_over_80 column')

    df['lim_over_4'] = df['bal_1_ex'] + df['bal_2_ex'] + df['bal_3_ex'] + df['bal_4_ex']

    df['credit_over_80'] = df['lim_over_4'] >= 4

    df['credit_over_80'] = df['credit_over_80'] * 1
    
    df = pd.concat([df, pd.get_dummies(df['pay_score_sums'], prefix='score')], axis = 1)
    
    df.drop(['lim_over_4', 'bal_1_ex', 'bal_2_ex', 'bal_3_ex', 'bal_4_ex', 'AGE', 'EDUCATION', 'pay_score_sum', 'pay_score_sums'], axis = 1, inplace = True) 

    print('Filtering features')
    
    feature_columns = df.columns
    
    target = df['default payment next month']
    features = df[feature_columns]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=34,test_size=0.25)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train =pd.DataFrame(data=scaler.transform(X_train), columns=feature_columns)
    X_test =pd.DataFrame(data=scaler.transform(X_test), columns=feature_columns)

    # Create correlation matrix
    corr_matrix = X_train.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Compute the correlation matrix
    corr = X_train.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_train.drop(columns=to_drop, inplace=True)
    X_test.drop(columns=to_drop, inplace=True)


    selector = SelectKBest(f_regression, k=18)

    selector.fit(X_train, y_train)

    selected_columns = X_train.columns[selector.get_support()]
    removed_columns = X_train.columns[~selector.get_support()]

    print('dropping columns')
    
    df.drop(removed_columns, axis=1)
    
    print('complete!')
    
    return df