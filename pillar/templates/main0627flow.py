from graphviz import Digraph

# 創建一個新的有向圖
flowchart = Digraph('Flowchart', node_attr={'shape': 'box'})

# 添加節點
flowchart.node('A', '設置日誌')
flowchart.node('B', '連接資料庫')
flowchart.node('C', '創建EPC執行緒')
flowchart.node('D', '創建車牌偵測執行緒')
flowchart.node('E', '啟動EPC執行緒')
flowchart.node('F', '啟動車牌偵測執行緒')
flowchart.node('G', '開始接收EPC')
flowchart.node('H', '處理接收到的EPC')
flowchart.node('I', 'EPC接收結束')
flowchart.node('J', '開始偵測車牌')
flowchart.node('K', '等待EPC執行緒結束')
flowchart.node('L', '等待車牌偵測執行緒結束')

# 添加邊
flowchart.edge('A', 'B')
flowchart.edge('B', 'C')
flowchart.edge('B', 'D')
flowchart.edge('C', 'E')
flowchart.edge('D', 'F')
flowchart.edge('E', 'G')
flowchart.edge('F', 'J')
flowchart.edge('G', 'H')
flowchart.edge('H', 'I')
flowchart.edge('I', 'K')
flowchart.edge('J', 'L')

# 保存流程圖到文件
flowchart.render('/Users/yunning_ben/Documents/pillar_ver.1.1/flow.png', format='png', cleanup=True)
