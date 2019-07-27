import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx
G = nx.karate_club_graph()#加载数据
cluster = list(greedy_modularity_communities(G))#聚类
colors=[]
#可视化
colorsSet=["g","r","b","w"]
for j in range(34):
    for i in range(len(cluster)):
        if j in cluster[i]:
            colors.append(colorsSet[i])
            break
nx.draw(G,node_color=colors,with_labels=True)
plt.show()

