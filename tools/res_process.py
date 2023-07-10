from collections import OrderedDict
import os
import json
import csv
import mmcv
from terminaltables import AsciiTable

def res_process(res_path,csv_root,in_domain=True):
        # Initialize the dictionary
    json_dict = {}
    seq_path = os.path.join(res_path,'res') 
    # Loop over all files in the path
    for file_name in os.listdir(seq_path):
        # Check if the file is a JSON file
        if file_name.endswith(".json"):
            # Extract the sequence ID from the file name
            seq_id = file_name.split(".")[0]
            # Read the JSON file
            with open(os.path.join(seq_path, file_name), "r") as f:
                content = json.load(f)
            # Append the content to the dictionary
            json_dict[seq_id] = content # json dict 

    # Print the dictionary
    # print(json_dict)
    seq_info_path = csv_root
    shift_type_dict = OrderedDict()
    shift_type_dict['all_shift_type'] = []
    seq_id_list = []
    with open(seq_info_path, 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header row
        for row in reader: 
            if (row[5]!='clear' or row[6]!='daytime') and in_domain:
                continue
            else:
                shift_type = row[3] 
                seq_id = row[0]
                if shift_type not in shift_type_dict:
                        shift_type_dict[shift_type] = []
                shift_type_dict[shift_type].append(seq_id)
                shift_type_dict['all_shift_type'].append(seq_id)
            # break
    # print(len(shift_type_dict['all_shift_type']))
    for type, seq_list in shift_type_dict.items():
        class_dict = OrderedDict()
        count_dict = OrderedDict()
        for seq_id in seq_list:
            eval_res = json_dict[seq_id]
            for i, item in enumerate(eval_res):
                if i ==0:
                    continue
                class_name = item[0]
                iou = item[1]
                acc = item[2]

                if acc != acc or iou != iou:
                    continue  # Ignore 0 or NaN values

                # Add IoU and Acc to the corresponding class sum
                if class_name not in class_dict:
                    class_dict[class_name] = [iou, acc]
                    count_dict[class_name] = 1
                else:
                    class_dict[class_name][0] += iou
                    class_dict[class_name][1] += acc
                    count_dict[class_name] += 1

        result_list = []
        result_list.append(['Class','IoU','Acc'])
        iou_sum = 0
        acc_sum = 0
        for class_name in class_dict:
            class_sum = class_dict[class_name]
            count = count_dict[class_name]
            mean_iou = class_sum[0] / count
            mean_acc = class_sum[1] / count
            iou_sum += mean_iou
            acc_sum += mean_acc
            
            result_list.append([class_name, round(mean_iou,2), round(mean_acc,2)])
        result_list.append(['*Average*',round(iou_sum/14,2),round(acc_sum/14,2)])
        save_name = 'evluation_indomain.txt' if in_domain else 'evluation.txt'
        table = AsciiTable(result_list)
        # Open a file for writing
        with open(os.path.join(res_path,save_name), 'a') as f:
            # Redirect the output to the file
            # print(table.table,'\n', file=f)
            f.write("Evaluation on "+type+'\n')
            f.write(table.table+'\n')
            

if __name__ == "__main__":    
    res_process('./work_dirs/')

