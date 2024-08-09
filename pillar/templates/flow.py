from graphviz import Digraph

# Create a new directed graph
flowchart = Digraph('Flowchart', node_attr={'shape': 'box'})

# Add nodes
flowchart.node('A', '主程式開啟')
flowchart.node('B', '辨識程式開啟')
flowchart.node('C', '偵測到車牌?')
flowchart.node('D', '確定並上傳至 \'license_records\' table')
flowchart.node('E', 'RFID接收器開啟')
flowchart.node('F', '掃描到標籤?')
flowchart.node('G', '上傳至 \'sensors\' table')
flowchart.node('H', '以時間和資料關聯性做配對並比較')

# Add edges
flowchart.edge('A', 'B')
flowchart.edge('A', 'E')
flowchart.edge('B', 'C')
flowchart.edge('C', 'D', label='是')
flowchart.edge('C', 'H', label='否', style='dashed')
flowchart.edge('D', 'H', style='dashed')
flowchart.edge('E', 'F')
flowchart.edge('F', 'G', label='是')
flowchart.edge('F', 'H', label='否', style='dashed')
flowchart.edge('G', 'H', style='dashed')

# Save the flowchart to a file
flowchart.render('/Users/yunning_ben/Documents/pillar_ver.1/flow.png', format='png', cleanup=True)
