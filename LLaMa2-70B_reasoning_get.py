
import os
import json
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  
def request_and_write(service, data_dir, args, model, tokenizer):
    service = service[2:-2]
    data_file_name = os.path.join(data_dir,service +"-"+ "train" +"-LLM-with_reasoning.json")
    #test_lines = json.load(open(data_file_name))
    with open(data_file_name, 'r') as json_file:
        test_lines = json.load(json_file)
    #print(len(test_lines))
    #sys.exit(1)
    flag = 0
    for idx_ in range(0, len(test_lines)):
        if idx_ == len(test_lines)-1:
            flag = -1
        turn_id = test_lines[idx_]['dialogue_id_turn'].split("-")[1]
        #print(f"turn id :{turn_id}")
        if int(turn_id) < args.request_turn:
            continue
        else:
            
            teacher_reasoning = test_lines[idx_]['reasoning']
            if teacher_reasoning != "In the given dialogue, the value of the requested slot is explicitly mentioned.":
                continue
        
        print(f"service: {service}, turn_id: {turn_id}")
        
        domain_slot = test_lines[idx_]['domain-slot']
        domain = domain_slot.split("-")[0].split("_")[0]
        slot = domain_slot.split("-")[1]
        request_text = "Performing the dialogue state tracking task. Consider the dialogue content: \""
        request_text = request_text + test_lines[idx_]['dialogue_content'] + "\","
        request_text = request_text + "the answer to the slot <" + domain+"-"+slot + "> is '" + test_lines[idx_]['groundtruth'] + "'.\n"
        request_text = request_text + "Can you tell me why and your reasoning process? In the context of dialogue state tracking, there are often multiple possible values associated with the requested slot. Please provide a concise explanation within six sentences of how to select the most appropriate value for the requested slot by carefully analyzing the dialogue context, user and system intent, and taking into consideration any confirmation or rejection information.\n"

        #request_data = "Performing the dialogue state tracking task. Consider the dialogue content: \"[USER]: I would like a Family Therapist in Novato. [SYSTEM]: There are 2 therapists. There is Jones Jeannette who is a Family Counselor in Novato. [USER]: Is there a therapist in El Cerrito? [SYSTEM]: There is 1. Rigg Christie is a Family Counselor in el Cerrito. [USER]: Can you book an appointment? [SYSTEM]: What time do you prefer? [USER]: I would like an appointment at 10:45 am. [SYSTEM]: When is the appointment for? [USER]: The appointment is for the 11th of March. [SYSTEM]: Booking appointment with Rigg Christie on March 11th at 10:45 am. [USER]: Can you make the appointment for the 5th of March instead? [SYSTEM]: Booking appointment with Rigg Christie on March 5th. [USER]: Yes, that sounds good. [SYSTEM]: Would you like to Rigg Christie on March 5th at 10:30 am instead? [USER]: No, that does not work for me.\", the answer to slot '<services-appointment_date>' is '10:45 am'.\n Can you tell me why and your reasoning process? In the context of dialogue state tracking, there are often multiple possible values associated with the requested slot. Please provide a concise explanation of how to select the most appropriate value for the requested slot by carefully analyzing the dialogue context, user and system intent, and taking into consideration any confirmation or rejection information.\n"
        #request_data = "Performing the dialogue state tracking task. Consider the dialogue content: \"[USER]: I want to get to 1150 Webster street. Can you help me find a taxi for 2 people?\",the answer to slot '<ridesharing-destination>' is '1150 webster street'.\n  Can you tell me why and your reasoning process? In the context of dialogue state tracking, there are often multiple possible values associated with the requested slot. Please provide a concise explanation of how to select the most appropriate value for the requested slot by carefully analyzing the dialogue context, user and system intent, and taking into consideration any confirmation or rejection information.\n"
        #print(request_text)
        #sys.exit(1)
        
        prompt = request_text
        prompt_template=f'''[INST] <<SYS>>
        Just return a concise reasoning process.
        <</SYS>>
        {prompt}[/INST]

        '''

        input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()

        for reasoning_id in range(args.reasoning_num):

            output = model.generate(inputs=input_ids, temperature=args.temperature, do_sample=True, top_p=0.9, top_k=40, max_new_tokens=512)
            answer = tokenizer.decode(output[0])
            if "</s>" in answer:
                answer = answer.replace("</s>","")
                
            test_lines[idx_]['reasoning_'+str(reasoning_id+1)] = answer.split("[/INST]")[-1].lstrip()

            print(f"Reasoning {reasoning_id+1}:")
            print(test_lines[idx_]['reasoning_'+str(reasoning_id+1)])
        print()
        test_lines[idx_]['reasoning'] = "done."
        if idx_ == len(test_lines)-1:
            flag = -1
        else:
            flag = idx_
        break
    
    with open(data_file_name, 'w') as json_file:
        json.dump(test_lines, json_file, indent=4)
    
    return flag
        
def main(args):
    
    dataset_order = ["['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                    "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                    "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                    "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
    
    
    data_dir = os.path.join("./data","SGD_single_service_train_teacher_data_multi-positive-samples")
    
    model_name_or_path = "/your_model_path/Llama-2-70B-chat-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model.eval()
    print(f"data_dir: {data_dir}")
    for service in dataset_order:
        while True:
            flag = request_and_write(service, data_dir, args, model, tokenizer)
            print(f"service: {service}，flag: {flag}")
            if flag == -1:
                 break
        #break

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--instruct", type=str, 
                        default="", help = "")
    parser.add_argument("--temperature", type=float, default=0.7, help="")
    parser.add_argument("--request_turn", type=int, default=10, help="")
    parser.add_argument("--reasoning_num", type=int, default=5, help="")

    args = parser.parse_args()
    main(args)
