import pandas as pd
import numpy as np

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
value=(value+10)/4
matrix = value[:1000, 1:]

#데이터 형식 변환
data=pd.DataFrame(matrix,columns=[np.arange(1,101)])
data2=np.zeros([100000,3])
#user id
u=0
for i in range(0,1000):
    data2[u:u+100,0]=str(data.index[i])
    u=u+100
#item id
k=1
for i in range(100000):
    data2[i,1]=k
    k+=1
    if k>100:
        k=1
#rating
k=0
for i in range(1000):
    for j in range(100):
        data2[k,2]=data.values[i,j]
        k=k+1
#dataframe으로 변환
df=pd.DataFrame(data2,columns=["uid","iid","rate"])
