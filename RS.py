from __future__ import (absolute_import,division,print_function,unicode_literals)
import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import KNNBasic
import os
from surprise import Reader
from collections import defaultdict

#Assignment_1
def load_surprise():
    data=Dataset.load_builtin('ml-100k')
    algo=SVD()
    cross_validate(algo,data,measures=['RMSE','MAE'],cv=5,verbose=True)

def open_dataset(filename):
    # jester파일 불러오기
    df1=pd.read_excel("jester-data-1.xls")

    #행추가 및 COLUMNS명 바꾸기
    C=df1.columns.tolist()
    df1.loc[24982]=C
    df1.columns=list(np.arange(1,102))

    #모두 rating한 user만 추출
    df2=df1[df1[1].isin([100])]

    #1000x100 matrix로 변환
    value = df2.values
    m=value[:1000,1:]
    value=(value+10)/20
    matrix = value[:1000, 1:]

    return matrix

def trans_matrix(matrix):
    #데이터 형식 변환
    num_i=len(matrix[0])
    num_u=len(matrix)
    
    data=pd.DataFrame(matrix,columns=[np.arange(1,num_i+1)])
    data2=np.zeros([num_u*num_i,3])
    #user id
    u=0
    for i in range(0,num_i):
        data2[u:u+num_i,0]=str(data.index[i])
        u=u+num_i
    #item id
    k=1
    for i in range(num_u*num_i):
        data2[i,1]=k
        k+=1
        if k>num_i:
            k=1
    #rating
    k=0
    for i in range(num_u):
        for j in range(num_i):
            data2[k,2]=data.values[i,j]
            k=k+1
    #dataframe으로 변환
    df=pd.DataFrame(data2,columns=["uid","iid","rate"])
    
    return df

#Assignment_2
#Similarity
def COS(a):
    NumUsers=np.size(a,axis=0)
    Sim=np.full((NumUsers,NumUsers),0.0) # similarity matrix 초기화
    
    for u in range(0,NumUsers):
        for v in range(u,NumUsers):
            InnerDot=np.dot(a[u,],a[v,])
            NormU=np.linalg.norm(a[u,])
            NormV=np.linalg.norm(a[v,])
            
            #예외처리
            if (NormU*NormV)==0: #모든 값이 0점일 경우 분모가 0이 됨
                Sim[u,v]=9999
                Sim[v,u]=9999
            else:
                Sim[u,v]=InnerDot/(NormU*NormV)
                Sim[v,u]=Sim[u,v]
    return Sim


def PCC(a):
    NumUsers=np.size(a,axis=0)
    Sim=np.full((NumUsers,NumUsers),0.0) #similarity matrix 초기화
    
    mean=np.nanmean(np.where(a!=0,a,np.nan),axis=1) #각 user당 평균 구해주기
    
    for u in range(0,NumUsers):
        for v in range(u,NumUsers):
            arridx_u=np.where(a[u,]==0)
            arridx_v=np.where(a[v,]==0)
            arridx=np.concatenate((arridx_u,arridx_v),axis=None)
            
            U=np.delete(a[u,],arridx)
            V=np.delete(a[v,],arridx)
            
            U=U-mean[u]
            V=V-mean[v]
            
            InnerDot=np.dot(U,V)
            NormU=np.linalg.norm(U)
            NormV=np.linalg.norm(V)
            
            #예외처리
            if (NormU*NormV)==0: #모든 값이 같을 경우 분모가 0이 됨
                Sim[u,v]=9999
                Sim[v,u]=9999
            else:
                Sim[u,v]=InnerDot/(NormU*NormV)
                Sim[v,u]=Sim[u,v]
    return Sim


def EUC(a):
    NumUsers = np.size(a, axis = 0)
    Sim = np.full((NumUsers, NumUsers), 0.0)
    for u in range(0, NumUsers):
        for v in range(u, NumUsers):
            tmp = np.sum(np.square(a[u, ] - a[v, ]))
            Sim[u, v] = np.sqrt(tmp)
            Sim[v, u] = Sim[u, v]
    return Sim



#Assignment3
def basic_baseline(mat,sim,k):
    predicted_rating=np.array([[0.0 for col in range(100)] for row in range(1000)])#예측 matrix
    bui=np.array([[0.0 for col in range(100)] for row in range(1000)]) #각 user의 bui matrix
    
    u_mean=np.nanmean(np.where(mat!=0,mat,np.nan),axis=1) #user평균
    i_mean=np.nanmean(np.where(mat!=0,mat,np.nan),axis=0) #item평균
    mean=sum(sum(mat))/100000 #모든 평균
    
    mat=np.array(mat,dtype=np.float64)
    
    if(sim=='COS'):
        Sim=COS(mat)
    elif(sim=='PCC'):
        Sim=PCC(mat)
    
    k_neighbors=np.argsort(-Sim)
    k_neighbors=np.delete(k_neighbors,np.s_[k+1:],1) #상위 k명을 추출함(자신도 포함되어있기 때문에 +1을 해줌)
    
    NumUsers=np.size(mat,axis=0) #1000
    NumItems=np.size(mat,axis=1) #100
    
    bi=i_mean-mean #bi구하기 
    bu=u_mean-mean #bu구하기
    
    #각 user의 bui값을 구해줌
    for u in range(0,NumUsers):
        for i in range(0,NumItems):
            bui[u][i]=mean+bu[u]+bi[i]
    
    #predicted rating matrix
    for u in range(0,NumUsers):
        list_sim=Sim[u,k_neighbors[u,1:]] #user u에 대한 상위 k명의 similarity를 가져옴 , 1: 자신을 제외한 k명
        list_rating=mat[k_neighbors[u,1:],].astype('float64') #상위 k명의 실제 rating값을 가져옴
        list_umean=u_mean[k_neighbors[u,1:],] #상위 k명의 user_mean을 가져옴
       #list_imean=i_mean[k_neighbors[u,],] #상위 k명의 item_mean을 가져옴
            
        denominator=np.sum(list_sim)
        numerator=np.sum(list_sim.reshape(-1,1)*((list_rating-bui[k_neighbors[u,1:],])),axis=0)
        predicted_rating[u]=bui[u]+(numerator/denominator)
        
    return predicted_rating

#Assigment4
def apply_to_surprise(df,form):
    num_u=len(df.groupby(df.uid).sum())
    num_i=len(df.groupby(df.iid).sum())
    
    reader=Reader(line_format='user item rating',sep='\t')
    data=Dataset.load_from_df(df,reader)
    
    from surprise import SVD
    algo=SVD()
    cross_validate(algo,data,measures=['RMSE','MAE'],cv=5,verbose=True)
    
    trainset=data.build_full_trainset()
    
    if form=='basic':
        sim_options={'name':'cosine','user_based':True}
        algo=KNNBasic(k=40,min_k=1,sim_options=sim_options)
        algo.fit(trainset)

        uid=1
        iids=df[df.uid==uid]

        for j in range(0,num_u):
            uid=j
            iids=df[df.uid==uid]
    
        for i in range(1,num_i+1):
            iid=iids[i-1:i].iid.values[0]
            r_ui=iids[i-1:i].rate.values[0]
            pred=algo.predict(uid,iid,r_ui,verbose=True)

    elif form=='CFwithmean':
        from surprise import KNNWithMeans
        
        sim_options={'name':'cosine','user_based':True}
        algo=KNNWithMeans(k=40,min_k=1,sim_options=sim_options)
        algo.fit(trainset)

        uid=0
        iids=df[df.uid==uid]

        for j in range(0,num_u):
            uid=j
            iids=df[df.uid==uid]
    
            for i in range(1,num_i+1):
                iid=iids[i-1:i].iid.values[0]
                r_ui=iids[i-1:i].rate.values[0]
                pred=algo.predict(uid,iid,r_ui,verbose=True)


    elif form=='CFwithz-score':
        from surprise import KNNWithZScore
        
        sim_options={'name':'cosine','user_based':True}
        algo=KNNWithZScore(k=40,min_k=1,sim_options=sim_options)
        algo.fit(trainset)

        uid=0
        iids=df[df.uid==uid]

        for j in range(0,num_u):
            uid=j
            iids=df[df.uid==uid]
    
            for i in range(1,num_i+1):
                iid=iids[i-1:i].iid.values[0]
                r_ui=iids[i-1:i].rate.values[0]
                pred=algo.predict(uid,iid,r_ui,verbose=True)

    elif form=='SVD':
        
        from surprise import SVD
        
        sim_options={'name':'cosine','user_based':True}
        algo=SVD(n_factors=100,n_epochs=20,biased=False,lr_all=0.005,reg_all=0)
        algo.fit(trainset)

        uid=0
        iids=df[df.uid==uid]

        for j in range(0,num_u):
            uid=j
            iids=df[df.uid==uid]
    
            for i in range(1,num_i+1):
                iid=iids[i-1:i].iid.values[0]
                r_ui=iids[i-1:i].rate.values[0]
                pred=algo.predict(uid,iid,r_ui,verbose=True)


    elif form=='PMF':
        from surprise import SVD
        
        sim_options={'name':'cosine','user_based':True}
        algo=SVD(n_factors=100,n_epochs=20,biased=False,lr_all=0.005,reg_all=0.02)
        algo.fit(trainset)

        uid=0
        iids=df[df.uid==uid]

        for j in range(0,num_u):
            uid=j
            iids=df[df.uid==uid]
    
            for i in range(1,num_i+1):
                iid=iids[i-1:i].iid.values[0]
                r_ui=iids[i-1:i].rate.values[0]
                pred=algo.predict(uid,iid,r_ui,verbose=True)


    elif form=='PMFwithBiased':
        from surprise import SVD
        
        sim_options={'name':'cosine','user_based':True}
        algo=SVD(n_factors=100,n_epochs=20,biased=True,lr_all=0.005,reg_all=0.02)
        algo.fit(trainset)

        uid=0
        iids=df[df.uid==uid]

        for j in range(0,num_u):
            uid=j
            iids=df[df.uid==uid]
    
            for i in range(1,num_i+1):
                iid=iids[i-1:i].iid.values[0]
                r_ui=iids[i-1:i].rate.values[0]
                pred=algo.predict(uid,iid,r_ui,verbose=True)


#Assignment5
def precision_recall_F1(df):
    #precision at k
    def precision_recall_at_k(predictions, k=10, threshold=3.5):
        '''Return precision and recall at k metrics for each user.'''

        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        return precisions, recalls

#precision / recall / F1 with surprise library
    from surprise.model_selection import KFold
    
    reader=Reader(line_format='user item rating',sep='\t')
    
    data=Dataset.load_from_df(df,reader)
    kf=KFold(n_splits=5)
    algo=SVD()

    for trainset,testset in kf.split(data):
        algo.fit(trainset)
        predictions=algo.test(testset)
        precisions, recalls=precision_recall_at_k(predictions,k=5,threshold=0.7)
    
        P=sum(prec for prec in precisions.values())/len(precisions)
        R=sum(rec for rec in recalls.values())/len(recalls)
        F1=2*P*R/(P+R)
    
        print("precision : ",P)
        print("recall : ",R)
        print("F1 : ",F1)


#NDCG

def NDCG(df,form):
    import math
    
    num_u=len(df.groupby(df.uid).sum())
    num_i=len(df.groupby(df.iid).sum())
    
    pre=pd.DataFrame(columns=['r','p'])#r:실제 / p:예측
    result=pd.DataFrame(columns=['u','NDCG']) # NDCG 결과 값
    
    reader=Reader(line_format='user item rating',sep='\t')
    data=Dataset.load_from_df(df,reader)
    
    if form=='basic':
        from surprise import KNNBasic
        trainset=data.build_full_trainset()
        sim_options={'name':'cosine','user_based':True}
        algo=KNNBasic(k=40,min_k=1,sim_options=sim_options)
        algo.fit(trainset)
    
    elif form=='CFwithmean':
        from surprise import KNNWithMeans
        trainset=data.build_full_trainset()
        sim_options={'name':'cosine','user_based':True}
        algo=KNNWithMeans(k=40,min_k=1,sim_options=sim_options)
        algo.fit(trainset)
        
    elif form=='CFwithz-score': 
        from surprise import KNNWithZScore
        trainset=data.build_full_trainset()
        sim_options={'name':'cosine','user_based':True}
        algo=KNNWithZScore(k=40,min_k=1,sim_options=sim_options)
        algo.fit(trainset)
        
    elif form=='SVD':
        from surprise import SVD
        trainset=data.build_full_trainset()
        algo=SVD(n_factors=100,n_epochs=20,biased=False,lr_all=0.005,reg_all=0)
        algo.fit(trainset)
        
    elif form=='PMF':
        from surprise import SVD
        trainset=data.build_full_trainset()
        algo=SVD(n_factors=100,n_epochs=20,biased=False,lr_all=0.005,reg_all=0.02)
        algo.fit(trainset)
        
    elif form=='PMFwithbias':
        from surprise import SVD
        trainset=data.build_full_trainset()
        algo=SVD(n_factors=100,n_epochs=20,biased=True,lr_all=0.005,reg_all=0.02)
        algo.fit(trainset)
    
    
    for j in range(0,num_u): #모든 user
        uid=j
        iids=df[df.uid==uid]
    
        for i in range(1,num_i+1): #모든 item
            iid=iids[i-1:i].iid.values[0]
            r_ui=iids[i-1:i].rate.values[0]
            pred=algo.predict(uid,iid,r_ui,verbose=True)
            pre.loc[i]=[pred.r_ui,pred.est] #pre dataframe에 실제 rating, 예측 rating을 넣음
    
        #TOP 10
        #pre_p : 예측r기준
        #pre_r : 실제r기준
        pre_p=pre.sort_values(by='p',ascending=False).head(10) #예측 rating을 기준으로 sorting후 top10개를 추출
        pre_r=pre_p.sort_values(by='r',ascending=False) #뽑힌10개를 실제 rating을 기준으로 다시 sorting
        pre_p=pre_p.set_index(np.arange(10)) #다시 인덱싱해줌 (밑에 계산해줄 때 사용)
        pre_r=pre_r.set_index(np.arange(10))
    
        #DCG
        DCG=pre_p['r'][0] # r[u,p1]
        for k in range(1,10): # r[u,pi]/log2(i)
            DCG+=pre_p['r'][k]/math.log(k+1,2)
    
        #IDCG
        IDCG=pre_r['r'][0]
        for k in range(1,10):
            IDCG+=pre_r['r'][k]/math.log(k+1,2)
    
        #NDCG
        NDCG=DCG/IDCG
        
        result.loc[j]=[j,NDCG]
        
    return result
                

#Final_Tern
#Singularity
def singularity(a):
    num_u=len(a)
    num_i=len(a[0])
    
    Sp=np.zeros([num_i]) ; Sn=np.zeros([num_i])
    Sp=1-(sum(a>0.5)/num_u)
    Sn=1-(sum(a<=0.5)/num_u)

    Sim=np.zeros([num_u,num_u])
    
    for u in range(num_u):
        for v in range(u,num_u):
            AA=np.zeros([num_i]) ; BB=np.zeros([num_i]) ; CC=np.zeros([num_i])
            U=10

            A=len(np.where((a[u,:]>0.5) & (a[v,:]>0.5))[0])
            index=np.where((a[u,:]>0.5) & (a[v,:]>0.5))
            AA[index]=AA[index]+(1-(a[u,index]-a[v,index])**2)*(Sp[index]**2)
            #print(AA[index])
            
            B=len(np.where((a[u,:]<=0.5) & (a[v,:]<=0.5))[0])
            index=np.where((a[u,:]<=0.5) & (a[v,:]<=0.5))
            BB[index]=BB[index]+(1-(a[u,index]-a[v,index])**2)*(Sn[index]**2)
            
            C=len(np.where(((a[u,:]>0.5) & (a[v,:]<=0.5)) | ((a[u,:]<=0.5) & (a[v,:]>0.5)))[0])
            index=np.where(((a[u,:]>0.5) & (a[v,:]<=0.5)) | ((a[u,:]<=0.5) & (a[v,:]>0.5)))
            CC[index]=CC[index]+(1-(a[u,index]-a[v,index])**2)*(Sp[index]*Sn[index])
            
            if A==0 or B==0 or C==0:
                Sim[u,v]=0
                Sim[v,u]=Sim[u,v]
            else:
                Sim[u,v]=(sum(AA)/A + sum(BB)/B + sum(CC)/C)/3
                Sim[v,u]=Sim[u,v]
                
    return Sim

#PIP
Rmax=1.0
Rmin=0.0
Rm=(Rmax+Rmin)/2

#Agreement
def Agreement(a,b):
    agreement=np.array(a*b>=Rm,dtype=np.int)
    agreement=np.where(agreement==0,2,agreement)
    #print(agreement)
    return agreement

#Proximity
def Proximity(a,u,v):
    proximity=np.zeros([len(a[0])])
    c=Agreement(a[u,:], a[v,:]) #True->1  /  False->2
    proximity=((2*(Rmax-Rmin)+1)-(c*np.abs(a[u,:]-a[v,:])))**2 
    return proximity

#impact
def Impact(a,u,v):
    impact=np.zeros([len(a[0])])
    agreement=Agreement(a[u,:],a[v,:])
    
    impact=np.where(agreement==1,(np.abs(a[u,:]-Rm)+1)*(np.abs(a[v,:]-Rm)+1),impact)
    impact=np.where(agreement==2,1/((np.abs(a[u,:]-Rm)+1)*(np.abs(a[v,:]-Rm)+1)),impact)
    #print(impact)
    return impact

#Popularity
def Popularity(a,u,v,avg):
    popularity=np.zeros([len(a[0])])
    popularity=np.where(((a[u,:]>avg) & (a[v,:]>avg)) | ((a[u,:]<avg) & (a[v,:]<avg)),1+((a[u,:]+a[v,:])/2-avg)**2,popularity)
    popularity=np.where(((a[u,:]>avg) & (a[v,:]<avg)) | ((a[u,:]<avg) & (a[v,:]>avg)),1,popularity)
    return popularity

#PIP
def PIP(a):
    n=len(a)
    PIP=np.zeros([n,n])
    
    avg=[] #아이템 평점의 평균을 구하는 시간을 줄이기 새로운 리스트를 할당하여 미리 구해놓은 뒤 활용
    for i in range (len(a[0])):
        avg.append(np.mean(a[:,i]))
        
    for u in range(n):
        for v in range(u, n):
            tmp=0.0
            tmp = Proximity(a,u,v) * Impact(a,u,v) * Popularity(a,u,v,avg)
            
            PIP[u,v]=sum(tmp)
            PIP[v,u]=PIP[u,v]
    return PIP