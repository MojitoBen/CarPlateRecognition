import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from graphviz import Digraph

def create_flowchart():
    dot = Digraph()

    # 定義節點
    dot.node('A', '開始 (Main)', shape='ellipse')
    dot.node('B', '設置日誌記錄', shape='box')
    dot.node('C', '連接到資料庫', shape='box')
    dot.node('D', 'Etag 接收執行緒', shape='parallelogram')
    dot.node('E', '車牌檢測執行緒', shape='parallelogram')
    dot.node('F', '更新配對執行緒', shape='parallelogram')
    dot.node('G', '開始接收 Etag', shape='box')
    dot.node('H', '開始檢測車牌', shape='box')
    dot.node('I', '更新車牌-傳感器配對', shape='box')
    dot.node('J', '插入 Etag', shape='box')
    dot.node('K', '插入車牌號碼', shape='box')
    dot.node('L', '更新 Etag 最後出現時間', shape='box')
    dot.node('M', '之後會把配對程式分出去', shape='box')
    dot.node('O', '資料庫操作', shape='box')
    dot.node('P', '更新車牌最後出現時間', shape='box')
    dot.node('Q', '更新配對', shape='box')

    # 定義邊
    dot.edges(['AB', 'BC'])
    dot.edge('C', 'D', '啟動 EPC 執行緒')
    dot.edge('C', 'E', '啟動車牌執行緒')
    dot.edge('C', 'F', '啟動更新配對執行緒')
    dot.edge('D', 'G', '運行 EPC 接收器')
    dot.edge('E', 'H', '運行車牌檢測器')
    dot.edge('F', 'I', '運行配對')
    dot.edge('G', 'L', '更新最後出現時間')
    dot.edge('G', 'J', '插入 Etag')
    dot.edge('H', 'K', '插入車牌信息')
    dot.edge('I', 'O', '插入/更新配對')
    dot.edge('H', 'P', '記錄車牌最後出現時間')
    dot.edge('I', 'Q', '記錄配對')

    dot.render('flowchart', format='png', cleanup=True)

create_flowchart()

# 顯示流程圖
img = mpimg.imread('flowchart.png')
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis('off')
plt.show()
