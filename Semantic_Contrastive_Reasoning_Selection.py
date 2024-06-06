from sentence_transformers import SentenceTransformer
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import sys
def dis(vec1, vec2):
    return numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))

request_turn = 10
dataset_order = ["['sgd_services_4']", "['sgd_flights_1']", "['sgd_services_3']",
                "['sgd_flights_3']", "['sgd_trains_1']", "['sgd_homes_2']", "['sgd_rentalcars_2']",
                "['sgd_restaurants_1']", "['sgd_music_1']", "['sgd_hotels_4']", "['sgd_media_2']",
                "['sgd_hotels_3']", "['sgd_rentalcars_3']", "['sgd_hotels_1']", "['sgd_homes_1']"]

data_dir1 = os.path.join("./data", "SGD_single_service_train_teacher_data")
data_dir2 = os.path.join("./data", "SGD_single_service_train_teacher_data_multi-positive-samples")
pdata_dir = os.path.join("./data", "SGD_single_service_train_teacher_data_multi-negative-samples")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
for service in dataset_order:
    service = service[2:-2]
    output_filename = "./embedding_data/" + service + "_embedding.json"
    output_data = []
    data_file_name1 = os.path.join(data_dir1,service +"-"+ "train" +"-LLM-with_reasoning.json")
    data_file_name2 = os.path.join(data_dir2,service +"-"+ "train" +"-LLM-with_reasoning.json")
    pdata_file_name = os.path.join(pdata_dir,service +"-"+ "train" +"-LLM-with_reasoning.json")
    with open(data_file_name1, 'r') as json_file:
        reasoning1_lines = json.load(json_file)
    with open(data_file_name2, 'r') as json_file:
        reasoning2_lines = json.load(json_file)
    with open(pdata_file_name, 'r') as json_file:
        preasoning_lines = json.load(json_file)
   
    assert len(reasoning1_lines) == len(reasoning2_lines) == len(preasoning_lines)

    # fig, ax = plt.subplots(figsize=(8,8),dpi = 600)
    # ax.axvline(x=0, color='black', linewidth=0.5)
    # ax.axhline(y=0, color='black', linewidth=0.5)
    for idx_ in range(0, len(reasoning1_lines)):
        turn_id = reasoning1_lines[idx_]['dialogue_id_turn'].split("-")[1]
        if int(turn_id) < request_turn:
            continue
        initial_reasoning = "In the given dialogue, the value of the requested slot is explicitly mentioned."
        if reasoning1_lines[idx_]['dialogue_id_turn'] == initial_reasoning or reasoning2_lines[idx_]['dialogue_id_turn'] == initial_reasoning or preasoning_lines[idx_]['dialogue_id_turn'] == initial_reasoning:
            print(f"reasoning error")
            sys.exit(1)
        
        print(f"{idx_}/{len(reasoning1_lines)}")
        item = {}
        examples = []
        domain_slot = reasoning1_lines[idx_]['domain-slot']
        domain = domain_slot.split("-")[0].split("_")[0]
        slot = domain_slot.split("-")[1]
        request_text = "Performing the dialogue state tracking task. Consider the dialogue content: \""
        request_text = request_text + reasoning1_lines[idx_]['dialogue_content'] + "\","
        request_text = request_text + "the answer to the slot <" + domain+"-"+slot + "> is '" + reasoning1_lines[idx_]['groundtruth'] + "'.\n"

        dialogue_context = request_text
        reasoning1 = reasoning1_lines[idx_]['reasoning']
        reasoning2 = reasoning2_lines[idx_]['reasoning_1']
        reasoning3 = reasoning2_lines[idx_]['reasoning_2']
        reasoning4 = reasoning2_lines[idx_]['reasoning_3']
        reasoning5 = reasoning2_lines[idx_]['reasoning_4']

        preasoning1 = preasoning_lines[idx_]['negative_reasoning_1']
        preasoning2 = preasoning_lines[idx_]['negative_reasoning_2']
        preasoning3 = preasoning_lines[idx_]['negative_reasoning_3']

        examples = [dialogue_context,reasoning1,reasoning2,reasoning3,reasoning4,reasoning5]
        examples = examples + [preasoning1, preasoning2, preasoning3]

        #print(examples)
        #sys.exit(1)
        
        embeddings = model.encode(examples)
        #print(embeddings)
        dc_emb = embeddings[0]
        negative_emb1 = embeddings[6]
        negative_emb2 = embeddings[7]
        negative_emb3 = embeddings[8]

        score_list = []
        T = 1


        for i in range(5):
            reasoning_vec = embeddings[i+1]
            score = (math.exp(dis(reasoning_vec, dc_emb) / T)) / (math.exp(dis(reasoning_vec, negative_emb1) / T) + math.exp(dis(reasoning_vec, negative_emb2) / T) + math.exp(dis(reasoning_vec, negative_emb3) / T))
            score_list.append(score)

        #print(f"score: {score_list}")
        #print(score_list.index(min(score_list)) )
        #sys.exit(1)
        #tsne2d = TSNE(n_components=2)
        pca2D = PCA(n_components=2)
        pca_2D = pca2D.fit_transform(embeddings)
        #vecs2d = tsne2d.fit_transform(embeddings)
        #print(pca_2D)

        x_ori = pca_2D[0][0]
        y_ori = pca_2D[0][1]

        for i in range(len(pca_2D)):
            pca_2D[i][0] = round((pca_2D[i][0] - x_ori), 4)
            pca_2D[i][1] = round((pca_2D[i][1] - y_ori), 4)

        
        item['DC'] = str([pca_2D[0][0], pca_2D[0][1]])
        item['R1'] = str([pca_2D[1][0], pca_2D[1][1]])
        item['R2'] = str([pca_2D[2][0], pca_2D[2][1]])
        item['R3'] = str([pca_2D[3][0], pca_2D[3][1]])
        item['R4'] = str([pca_2D[4][0], pca_2D[4][1]])
        item['R5'] = str([pca_2D[5][0], pca_2D[5][1]])
        item['PR1'] = str([pca_2D[6][0], pca_2D[6][1]])
        item['PR2'] = str([pca_2D[7][0], pca_2D[7][1]])
        item['PR3'] = str([pca_2D[8][0], pca_2D[8][1]])
        item['flag'] = "R"+str(score_list.index(min(score_list))+1)
        output_data.append(item)
        #break
        # ax.scatter(pca_2D[0][0], pca_2D[0][1], marker= 'p', label='Dialogue Content')

        # ax.scatter(pca_2D[1][0], pca_2D[1][1], marker= 'o', label='Reasoning')
        # ax.scatter(pca_2D[2][0], pca_2D[2][1], marker= 'o', label='Reasoning')
        # ax.scatter(pca_2D[3][0], pca_2D[3][1], marker= 'o', label='Reasoning')
        # ax.scatter(pca_2D[4][0], pca_2D[4][1], marker= 'o', label='Reasoning')
        # ax.scatter(pca_2D[5][0], pca_2D[5][1], marker= 'o', label='Reasoning')

        # ax.scatter(pca_2D[6][0], pca_2D[6][1], marker= '^', label='Perturbed Reasoning')
        # ax.scatter(pca_2D[7][0], pca_2D[7][1], marker= '^', label='Perturbed Reasoning')
        # ax.scatter(pca_2D[8][0], pca_2D[8][1], marker= '^', label='Perturbed Reasoning')

        # pca2D_df = pd.DataFrame(data =  pca_2D, columns = ['x', 'y'])
        # pca2D_df['cluster'] = ['DC','R','R','R','R','R','PR','PR','PR' ]
        # pca2D_df['style'] = ['1','2','2','2','2','2','3','3','3' ]
        # sns.scatterplot(x='x', y='y', hue='cluster', style="style", data=pca2D_df)
        
    #plt.title("PCA")
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_xticks(range(-1, 1, 10))
    # ax.set_yticks(range(-1, 1, 10))
    # #plt.show()
    # ax.legend()
    # plt.savefig('./Fig/1.png', dpi=600 ,bbox_inches='tight')
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=4)   
    #sys.exit(1)