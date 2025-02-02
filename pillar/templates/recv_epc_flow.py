from graphviz import Digraph

flowchart = Digraph('Flowchart', node_attr={'shape': 'box'})

flowchart.node('start', '開始')
flowchart.node('establish_connection', '建立 RFID 連接')
flowchart.node('send_start_command', '發送開始接收 EPC 數據的命令')
#loop
flowchart.node('loop', '進入無限循環')
flowchart.node('receive_data', '接收 EPC 數據')
flowchart.node('process_data', '處理接收到的 EPC 數據')
#check
flowchart.node('check_epc_in_dict', '如果 EPC 在 epc_dict 中:')
flowchart.node('check_time_interval', '檢查時間間隔是否超過')
flowchart.node('add_to_dict', '將當前的 EPC 添加到 epc_dict 中')
flowchart.node('update_last_seen_time', '更新最後一次出現時間')
flowchart.node('increase_count', '增加 EPC 在 epc_dict 中的計數')
flowchart.node('check_max_size', '如果 epc_dict 的大小超過最大字典大小:')
flowchart.node('remove_oldest_epc', '刪除 epc_dict 中的最舊的 EPC 記錄')
flowchart.node('update_time_record', '更新 epc_time_dict 中的時間記錄')
#interrupt
flowchart.node('keyboard_interrupt', '收到鍵盤中斷信號時:')
flowchart.node('send_stop_command', '發送停止接收數據的命令')
flowchart.node('close_connections', '關閉 RFID 連接和數據庫連接')
#end
flowchart.node('end', '結束')
# lines
flowchart.edge('start', 'establish_connection', label='建立 RFID 連接')
flowchart.edge('establish_connection', 'send_start_command', label='發送開始命令')
flowchart.edge('send_start_command', 'loop', label='進入循環')
flowchart.edge('loop', 'receive_data', label='接收 EPC 數據')
flowchart.edge('receive_data', 'process_data', label='處理 EPC 數據')
flowchart.edge('process_data', 'check_epc_in_dict', label='檢查 EPC 字典')
flowchart.edge('check_epc_in_dict', 'check_time_interval', label='檢查時間間隔')
flowchart.edge('check_time_interval', 'update_last_seen_time', label='更新最後出現時間', headlabel='是')
flowchart.edge('check_time_interval', 'increase_count', label='增加計數', headlabel='否')
flowchart.edge('update_last_seen_time', 'loop', label='返回循環')
flowchart.edge('increase_count', 'loop', label='返回循環')
flowchart.edge('check_epc_in_dict', 'add_to_dict', label='添加到字典', headlabel='否')
flowchart.edge('add_to_dict', 'loop', label='返回循環')
flowchart.edge('check_max_size', 'remove_oldest_epc', label='刪除最舊 EPC 記錄', headlabel='是')
flowchart.edge('remove_oldest_epc', 'update_time_record', label='更新時間記錄')
flowchart.edge('update_time_record', 'loop', label='返回循環')
flowchart.edge('check_max_size', 'loop', label='返回循環', headlabel='否')
flowchart.edge('loop', 'keyboard_interrupt', label='檢查鍵盤中斷信號')
flowchart.edge('keyboard_interrupt', 'send_stop_command', label='發送停止命令')
flowchart.edge('send_stop_command', 'close_connections', label='關閉連接')
flowchart.edge('close_connections', 'end', label='結束')

flowchart.render('flowchart', format='png', cleanup=True)
