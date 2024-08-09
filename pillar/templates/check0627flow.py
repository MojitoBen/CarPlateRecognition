from graphviz import Digraph

# 創建一個新的有向圖
flowchart2 = Digraph('Flowchart2', node_attr={'shape': 'box'})

# 添加節點
flowchart2.node('A', '設置日誌')
flowchart2.node('B', '連接資料庫')
flowchart2.node('C', '開始處理重複綁定 (迴圈)')
flowchart2.node('D', '檢查現有組群')
flowchart2.node('E', '更新或插入新組群')
flowchart2.node('F', '檢查重複綁定是否存在')
flowchart2.node('G', '插入重複綁定')

# 添加邊
flowchart2.edge('A', 'B')
flowchart2.edge('B', 'C')
flowchart2.edge('C', 'D')
flowchart2.edge('D', 'E')
flowchart2.edge('E', 'F')
flowchart2.edge('F', 'G')
flowchart2.edge('G', 'C', label='迴圈', style='dashed')

# 保存流程圖到文件
flowchart2.render('/Users/yunning_ben/Documents/pillar_ver.1.1/checked.png', format='png', cleanup=True)
