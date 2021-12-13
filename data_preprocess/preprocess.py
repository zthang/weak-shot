import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm


def cosine_distance(x, y):
    feature1 = F.normalize(x, dim=0)
    feature2 = F.normalize(y, dim=0)
    distance = 1 + (feature1 * feature2).sum(dim=0)
    return distance/2

def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min)/(max-min)

def get_dataset(prefix):
    base_train_image = get_image_embedding(f"{prefix}image_embed_base_train.pkl")
    base_test_image = get_image_embedding(f"{prefix}image_embed_base_test.pkl")
    # novel_train_image = get_image_embedding(f"{prefix}image_embed_novel_train.pkl")
    # novel_test_image = get_image_embedding(f"{prefix}image_embed_novel_test.pkl")
    base_train_label = pickle.load(open(f"{prefix}image_category_base_train.pkl", "rb"))
    base_test_label = pickle.load(open(f"{prefix}image_category_base_test.pkl", "rb"))
    # novel_train_label = pickle.load(open(f"{prefix}image_category_novel_train.pkl", "rb"))
    # novel_test_label = pickle.load(open(f"{prefix}image_category_novel_test.pkl", "rb"))

    # return base_train_image, base_test_image, novel_train_image, novel_test_image, base_train_label, base_test_label, novel_train_label, novel_test_label
    return base_train_image, base_test_image, base_train_label, base_test_label

def merge_image_embedding(prefix, type, num):
    image_embedding = None
    for i in range(num):
        embedding = pickle.load(open(f"{prefix}image_embed_{type}_{i}", "rb"))
        if image_embedding != None:
            image_embedding = torch.cat((image_embedding, embedding), dim=0)
        else:
            image_embedding = embedding
    pickle.dump(image_embedding, open(f"{prefix}image_embed_{type}.pkl", "wb"))
    print(f"{type} done.")

def get_image_embedding(filename):
    return pickle.load(open(filename, "rb"))

def make_graph_file(name, images):
    print(name)
    mean = torch.mean(images, dim=0)
    images = images - mean
    f = open(f"graph/{name}.txt", "w")
    for source_node_id in range(images.size(0)):
        print(source_node_id)
        for target_node_id in range(source_node_id+1, images.size(0)):
            distance = cosine_distance(images[source_node_id], images[target_node_id])
            # distance = 1/torch.norm(images[source_node_id] - images[target_node_id], p=2)
            f.write(f"{source_node_id} {target_node_id} {distance.item()}\n")
    f.close()

def save_image_embedding(simnet, prefix, type, data_loader):
    image_category = None
    for batch_i, image_info in tqdm(enumerate(data_loader)):
        images, categories, file_names = image_info
        images = simnet.backbone(images)
        pickle.dump(images, open(f"{prefix}image_embed_{type}_{batch_i}", "wb"))
        if image_category != None:
            image_category = torch.cat((image_category, categories))
        else:
            image_category = categories
    pickle.dump(image_category, open(f"{prefix}image_category_{type}.pkl", "wb"))
    merge_image_embedding(prefix, type, len(data_loader))


