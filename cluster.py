import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

MAXDISTANCE=100   #规定最大距离，用于填补自环路距离，防止自己和自己聚类

#定义距离矩阵，并初始化
def distanceMatrix(cluster,adj_matrix):
    lenC=len(cluster)
    matrix=np.zeros([lenC,lenC])
    for i in range(lenC):
        for j in range(i,lenC):
            matrix[i,j]=setDistance(cluster[i],cluster[j],adj_matrix)
            matrix[j,i]=matrix[i,j]
    return matrix

#计算两个项的距离
def distance(A,B):
    sum=0#共同好友数
    all=0#总好友数
    for i in range(len(A)):
        if A[i]==1 and B[i]==1:
            sum+=1
        if A[i]==1 or B[i]==1:
            all+=1
    distance=1-sum/all
    return distance if distance!=0 else MAXDISTANCE#消除自回路的影响

#计算两个类的距离
def setDistance(A,B,adj_matrix):
    averageLinkage=0#距离和
    n=0#总数
    for i in A:
        for j in B:
            averageLinkage+=distance(adj_matrix[i-1],adj_matrix[j-1])
            n+=1
    return averageLinkage/n

#更新距离矩阵聚类结果列表
def updata(distMat,delete,cluster,adj_matrix):
    distMat=np.delete(distMat, delete[1], axis=0)#删除距离矩阵合并的行
    distMat=np.delete(distMat, delete[1], axis=1)#删除距离矩阵合并的列
    for i in cluster[delete[1]]:#改变聚类结果列表
        cluster[delete[0]].append(i)
    cluster[delete[0]].sort()
    cluster.remove(cluster[delete[1]])
    for i in range(len(cluster)):#更新距离
        distMat[i,delete[0]]=setDistance(cluster[i],cluster[delete[0]],adj_matrix)
        distMat[delete[0],i]=distMat[i,delete[0]]
    return distMat,cluster

#计算模块化度量指标
def modularity(G,cluster):
    numE=len(G.edges)
    numC=len(cluster)
    matrix=np.zeros([numC,numC])#定义矩阵
    for edge in G.edges:#查找边是属于那个类的，并计算矩阵
        edge=[int(edge[i]) for i in range(2)]
        sourse=MAXDISTANCE
        target=MAXDISTANCE
        for i in range(numC):
            if edge[0] in cluster[i]:
                sourse=i
            if edge[1] in cluster[i]:
                target=i
        if sourse!=MAXDISTANCE and target!=MAXDISTANCE:
            matrix[sourse,target]+=1
    for i in range(numC):#平分跨类的边
        for j in range(i,numC):
            sum=matrix[i,j]+matrix[j,i]
            matrix[i,j]=sum/(2*numE)
            matrix[j,i]=sum/(2*numE)
    eii=0
    a2=0
    for i in range(numC):
        eii+=matrix[i,i]
    for i in range(numC):
        a2+=matrix[i].sum()**2
    Q=eii-a2
    return Q

#层次聚类
def hierarchicalClustering(G):
    adj_matrix = nx.to_numpy_matrix(G)#得到邻接矩阵
    for i in range(34):
        adj_matrix[i,i]=1#建立自回路，使得互相认识的人不会增加距离
    adj_matrix=np.array(adj_matrix)
    cluster=[[int(i)] for i in G.nodes]#聚类结果列表
    distMat=distanceMatrix(cluster,adj_matrix)#距离矩阵
    maxModularity=0#最大Q值
    ret=[]
    while True:
        where=np.where(distMat==np.min(distMat))#寻找最小距离
        delete = [where[0][0],where[1][0]]#记录要合并的类
        delete.sort()
        distMat,cluster = updata(distMat,delete,cluster,adj_matrix)#更新距离矩阵和聚类结果列表
        modul = modularity(G,cluster)#计算Q
        if modul>maxModularity:#如果Q值最大，记录聚类结果列表
            ret=[]
            for i in cluster:
                ret.append(list(i))
            maxModularity=modul
        if len(cluster)==2:#聚类结束判定
            ret=[]
            for i in cluster:
                ret.append(list(i))
            break
    return ret

#生成可视化颜色列表
def color(cluster):
    colors=[]
    colorsSet=["g","r","b","w"]
    cluster=hierarchicalClustering(G)
    for j in range(34):
        for i in range(len(cluster)):
            if j+1 in cluster[i]:
                colors.append(colorsSet[i])
                break
    return colors
if __name__ == '__main__':
    G=nx.read_gml("karate.gml")
    cluster=hierarchicalClustering(G)
    colors=color(cluster)
    nx.draw(G,node_color=colors,with_labels=True)
    plt.show()