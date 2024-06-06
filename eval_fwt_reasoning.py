
import os
import sys
import json
from glob import glob
import argparse
from utils.dataset_order import get_dataset_order

def main(args):
    dataset_order = get_dataset_order(args.dataset_id)
            
    output_dir = os.path.join("./output", args.test_data_name)
    if not os.path.exists(output_dir):
        print(f"results dir {output_dir} not find!")
        sys.exit(1)
    
    JGA_list = []
    print("Calculating JGA score for each service.....")
    for service_id in range(1, len(dataset_order)):
        
        result_file = os.path.join(output_dir, str(service_id)+"-"+dataset_order[service_id] +"_result.txt")        
        if not os.path.exists(output_dir):
            print(f"result_file {result_file} not find!")
            sys.exit(1)
        model_results = open(result_file, "r").readlines()
        
        testfile_idx = "./data/SGD_single_service_test/" + dataset_order[service_id] + "-test.idx"
        testfile_name = "./data/SGD_single_service_test/" + dataset_order[service_id] + "-test-LLM.json"
        idx_lines = open(testfile_idx).readlines()
        test_lines = json.load(open(testfile_name))
        
        assert len(model_results) == len(idx_lines) == len(test_lines), "line number error!"
        
        dial_dic = {}
        # dia_dic['state'] = {}
        # dia_dic['pred_state'] = {}
        error_state = 0
        for idx_ in range(0, len(idx_lines)):
            true_state = test_lines[idx_]['output']
            result_line = model_results[idx_].strip().lower()
            idx_line = idx_lines[idx_].strip()
            if idx_line not in result_line:
                
                print(idx_line,result_line )
                sys.exit(1)
            pred_state = result_line.split("|||")[-1]
            pred_state = eval(pred_state)[0]
            
            
            # new 
            if "result:" in pred_state:
                pred_state = pred_state.split("result:")[-1].strip()
            elif "is the most appropriate value for the slot" in pred_state:
                #print("1-------------------------------------")
                #print(pred_state)
                pred_state = pred_state.split("is the most appropriate value for the slot")[0].strip()
                pred_state = pred_state.split(" ")[-1].strip()
                #print(pred_state)
                #sys.exit(1)
            elif "the most appropriate value for the" in pred_state:
                #print(f"2-------------------------------------{dataset_order[service_id]},{idx_}")
                #print(pred_state)
                pred_state = pred_state.split("the most appropriate value for the",1)[-1].strip()
               # print(pred_state)
                if " is " in pred_state:
                    pred_state = pred_state.split(" is ")[1].strip()
                elif " to " in pred_state:
                    pred_state = pred_state.split(" to ")[1].strip()
                else:
                    #print(pred_state)
                    #print("new paradim ")
                    error_state += 1
                    #pred_state = "none"
                
                #print(pred_state)
                if "," in pred_state:
                    pred_state = pred_state.split(",", 2)[0].strip()
                
                #print(pred_state)
                #sys.exit(1)
            elif "be filled with the value" in pred_state:
                #print("-------------------------------------")
                #print(pred_state)
                pred_state = pred_state.split("be filled with the value")[-1].strip()
                #print(pred_state)
                #print(pred_state)
                #sys.exit(1)
            elif "the answer to the slot" in pred_state:
                #print("-------------------------------------")
                #print(pred_state)
                pred_state = pred_state.split("the answer to the slot")[-1].strip()
                #print(pred_state)
                pred_state = pred_state.split(" is ")[1].strip()
                pred_state = pred_state.split(" ")[0]
                #print(pred_state)
                #sys.exit(1)
            elif "the most appropriate value is" in pred_state:
                pred_state = pred_state.split("the most appropriate value is")[-1].strip()
                #print(pred_state)
                #print()
                if "'" in pred_state:
                    pred_state = pred_state.split("'",2)[1]
                    #print(pred_state)
                else:
                    print(f" line {idx_}")
                    sys.exit(1)
            else:

                error_state += 1
                #sys.exit(1)
            
            
            
            if "</s>" in pred_state:
                pred_state = pred_state.replace("</s>","")
            
            dial_json_n, dial_idx, turn_idx, frame_idx, d_name, s_name = idx_line.split("|||") 
            dic_key_name = dial_idx + "-" + turn_idx
            if dic_key_name not in dial_dic:
                dial_dic[dic_key_name] = {}
                dial_dic[dic_key_name]['state'] = {}
                dial_dic[dic_key_name]['pred_state'] = {}
                dial_dic[dic_key_name]['state'][d_name + "-" + s_name] = true_state
                dial_dic[dic_key_name]['pred_state'][d_name + "-" + s_name] = pred_state
            else:
                dial_dic[dic_key_name]['state'][d_name + "-" + s_name] = true_state
                dial_dic[dic_key_name]['pred_state'][d_name + "-" + s_name] = pred_state
        # with open("pred.json", 'w') as f:
        #         json.dump(dial_dic, f, indent=4)   

        joint_total = 0
        joint_acc = 0
        for turn_id in dial_dic:
            joint_total += 1
            true_state_dic = dial_dic[turn_id]['state']
            pred_state_dic = dial_dic[turn_id]['pred_state']
            true_flag = 1
            for ds_key in true_state_dic:
                assert ds_key in pred_state_dic, "key find error"
                if true_state_dic[ds_key] != pred_state_dic[ds_key]:
                    true_flag = 0
                    break
            joint_acc = joint_acc + true_flag
        
        joint_accuracy = joint_acc / joint_total
        print('{}: {} JGA: {}'.format(service_id, dataset_order[service_id], joint_accuracy))
        JGA_list.append(joint_accuracy)
        #break    
    print(f"average JGA is {sum(JGA_list) / len(JGA_list)}")
    print()
    
    average_JGA = sum(JGA_list) / len(JGA_list)
    
    JGA_list.append(average_JGA)
    
    del(dataset_order[0])
    dataset_order.append("Average")
    import pandas as pd
    dataframe = pd.DataFrame({'service_name':dataset_order,'JGA score':JGA_list})
    
    dataframe.to_csv("./csv_files/reasoning_fwt_dataset_id_"+str(args.dataset_id)+".csv",index=True)
          
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default=5, type=int)

    #parser.add_argument("--test_data_name", type=str, default="t5small_Reasoning_LLaMa2-70B_dataset_id_5_fwt", help = "")
    parser.add_argument("--test_data_name", type=str, default="t5small_Reasoning_LLaMa2-70B_dataset_id_5_fwt_with_memoryreplay", help = "")

    args = parser.parse_args()
    main(args)
