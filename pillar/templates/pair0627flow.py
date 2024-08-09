from graphviz import Digraph

# 創建一個新的有向圖
flowchart1 = Digraph('Flowchart1', node_attr={'shape': 'box'})

# 添加節點
flowchart1.node('A', '設置日誌')
flowchart1.node('B', '連接資料庫')
flowchart1.node('C', '開始配對車牌感測器(定時)')
flowchart1.node('D', '查詢特定時間範圍內的車牌與etag標籤')
flowchart1.node('E', '插入配對到 plate_sensor_mappings 表')
flowchart1.node('F', '查詢唯一配對的車牌與標籤')
flowchart1.node('G', '更新或插入唯一的車牌與標籤到 unique_plate_code_bindings 表')
flowchart1.node('H', '查詢重複的車牌或標籤配對')
flowchart1.node('I', '插入重複配對到 duplicate_bindings 表')

# 添加邊
flowchart1.edge('A', 'B')
flowchart1.edge('B', 'C')
flowchart1.edge('C', 'D')
flowchart1.edge('D', 'E')
flowchart1.edge('E', 'F')
flowchart1.edge('F', 'G')
flowchart1.edge('G', 'H')
flowchart1.edge('H', 'I')
flowchart1.edge('I', 'C', label='定時檢查', style='dashed')

# 保存流程圖到文件
flowchart1.render('/Users/yunning_ben/Documents/pillar_ver.1.1/flow_mappings.png', format='png', cleanup=True)
