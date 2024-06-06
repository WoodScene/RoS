
import os
import json
import copy
from glob import glob
import argparse
import openai
import time
import sys

openai.api_key = "your_api_key"

class ChatGPT:
    def __init__(self,conversation_list=[], instruct = "", temperature = 0.5) -> None:
        self.instruct = instruct
        self.temperature = temperature
        self.conversation_list = [{'role':'system','content': self.instruct}]
        print(f"\U0001f47D: {self.instruct}\n")
        #self.conversation_list = []
        

    def show_conversation(self,msg_list):
        for msg in msg_list[-2:]:
            if msg['role'] == 'user':
                print(f"\U0001f47b(User): {msg['content']}\n")
            else:
                print(f"\U0001F9D9\U0000200D\U00002640\U0000FE0F(System): {msg['content']}\n")


    def retry_request(self, max_retries):
        retries = 0
        while retries < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    #model = "text-davinci-003",
                    messages=self.conversation_list,
                    temperature = self.temperature,
                    max_tokens = 256,
                    )
                return response
            except Exception as e:
                if e:
                    print("Request failed:", e)
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                    retries += 1
                else:
                    raise e
        print("Maximum number of retries reached. Request failed.")
        return None


    def ask(self,prompt):
        self.conversation_list.append({"role":"user","content":prompt})
        
        response = self.retry_request(10)
        
        answer = response.choices[0].message['content']
        self.conversation_list.append({"role":"assistant","content":answer})
        self.show_conversation(self.conversation_list)
        
        return answer
        
def request_and_write(service, data_dir, args):
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
        request_text = request_text + "Can you tell me why and your reasoning process? In the context of dialogue state tracking, there are often multiple possible values associated with the requested slot. Please provide a concise explanation of how to select the most appropriate value for the requested slot by carefully analyzing the dialogue context, user and system intent, and taking into consideration any confirmation or rejection information.\n"

        #request_data = "Performing the dialogue state tracking task. Consider the dialogue content: \"[USER]: I would like a Family Therapist in Novato. [SYSTEM]: There are 2 therapists. There is Jones Jeannette who is a Family Counselor in Novato. [USER]: Is there a therapist in El Cerrito? [SYSTEM]: There is 1. Rigg Christie is a Family Counselor in el Cerrito. [USER]: Can you book an appointment? [SYSTEM]: What time do you prefer? [USER]: I would like an appointment at 10:45 am. [SYSTEM]: When is the appointment for? [USER]: The appointment is for the 11th of March. [SYSTEM]: Booking appointment with Rigg Christie on March 11th at 10:45 am. [USER]: Can you make the appointment for the 5th of March instead? [SYSTEM]: Booking appointment with Rigg Christie on March 5th. [USER]: Yes, that sounds good. [SYSTEM]: Would you like to Rigg Christie on March 5th at 10:30 am instead? [USER]: No, that does not work for me.\", the answer to slot '<services-appointment_date>' is '10:45 am'.\n Can you tell me why and your reasoning process? In the context of dialogue state tracking, there are often multiple possible values associated with the requested slot. Please provide a concise explanation of how to select the most appropriate value for the requested slot by carefully analyzing the dialogue context, user and system intent, and taking into consideration any confirmation or rejection information.\n"
        #request_data = "Performing the dialogue state tracking task. Consider the dialogue content: \"[USER]: I want to get to 1150 Webster street. Can you help me find a taxi for 2 people?\",the answer to slot '<ridesharing-destination>' is '1150 webster street'.\n  Can you tell me why and your reasoning process? In the context of dialogue state tracking, there are often multiple possible values associated with the requested slot. Please provide a concise explanation of how to select the most appropriate value for the requested slot by carefully analyzing the dialogue context, user and system intent, and taking into consideration any confirmation or rejection information.\n"
        #print(request_text)
        #sys.exit(1)
        for reasoning_id in range(args.reasoning_num):

            chat = ChatGPT(instruct= args.instruct, temperature=args.temperature)
            answer = chat.ask(request_text)
            
            test_lines[idx_]['reasoning_'+str(reasoning_id+1)] = answer

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
    dataset_order = [ "['sgd_flights_1']", "['sgd_services_3']",
                    "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                    "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                    "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]
    data_dir = os.path.join("./data","SGD_single_service_train_ChatGPT-reasoning_data_multi-positive-samples")
    print(f"data_dir: {data_dir}")
    for service in dataset_order:
        while True:
            flag = request_and_write(service, data_dir, args)
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
    parser.add_argument("--reasoning_num", type=int, default=5, help="") # 5个正样本

    args = parser.parse_args()
    main(args)
