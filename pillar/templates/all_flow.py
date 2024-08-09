from graphviz import Digraph

# 創建一個新的有向圖
flowchart = Digraph('Flowchart', node_attr={'shape': 'box'})

# 添加節點
# 第一步設置日誌
flowchart.node('A', '設置日誌')

# 連接資料庫
flowchart.node('B', '連接資料庫')

# EPC執行緒節點
flowchart.node('C', '創建EPC執行緒')
flowchart.node('D', '啟動EPC執行緒')
flowchart.node('E', '開始接收EPC')
flowchart.node('F', '處理接收到的EPC')
flowchart.node('G', 'EPC接收結束')
flowchart.node('H', '等待EPC執行緒結束')

# 車牌偵測執行緒節點
flowchart.node('I', '創建車牌偵測執行緒')
flowchart.node('J', '啟動車牌偵測執行緒')
flowchart.node('K', '開始偵測車牌')
flowchart.node('L', '等待車牌偵測執行緒結束')

# 配對車牌與標籤節點
flowchart.node('M', '開始配對車牌感測器(定時)')
flowchart.node('N', '查詢特定時間範圍內的車牌與標籤')
flowchart.node('O', '插入配對到 plate_sensor_mappings 表')
flowchart.node('P', '查詢唯一配對的車牌與標籤')
flowchart.node('Q', '更新或插入唯一的車牌與標籤到 unique_plate_code_bindings 表')
flowchart.node('R', '查詢重複的車牌或標籤配對')
flowchart.node('S', '插入重複配對到 duplicate_bindings 表')

# 檢查重複綁定節點
flowchart.node('T', '開始處理重複綁定 (迴圈)')
flowchart.node('U', '檢查現有組群')
flowchart.node('V', '更新或插入新組群')
flowchart.node('W', '檢查重複綁定是否存在')
flowchart.node('X', '插入重複綁定')

# 添加邊
flowchart.edge('A', 'B')

# EPC執行緒流程
flowchart.edge('B', 'C')
flowchart.edge('C', 'D')
flowchart.edge('D', 'E')
flowchart.edge('E', 'F')
flowchart.edge('F', 'G')
flowchart.edge('G', 'H')

# 車牌偵測執行緒流程
flowchart.edge('B', 'I')
flowchart.edge('I', 'J')
flowchart.edge('J', 'K')
flowchart.edge('K', 'L')

# 配對車牌與標籤流程
flowchart.edge('B', 'M')
flowchart.edge('M', 'N')
flowchart.edge('N', 'O')
flowchart.edge('O', 'P')
flowchart.edge('P', 'Q')
flowchart.edge('Q', 'R')
flowchart.edge('R', 'S')
flowchart.edge('S', 'M', label='定時檢查', style='dashed')

# 檢查重複綁定流程
flowchart.edge('B', 'T')
flowchart.edge('T', 'U')
flowchart.edge('U', 'V')
flowchart.edge('V', 'W')
flowchart.edge('W', 'X')
flowchart.edge('X', 'T', label='迴圈', style='dashed')

# 保存流程圖到文件
flowchart.render('/Users/yunning_ben/Documents/pillar_ver.1.1/combined_flowchart.png', format='png', cleanup=True)
