import string


from utils import load_file


def extracting_captions(data):
    captions = {}
    for caption in data.split("\n"):
        image_descriptions = caption.split()
        if len(caption) < 2:
            continue
        image_id = image_descriptions[0].split(".")[0]
        image_description = " ".join(image_descriptions[1:])
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(image_description)
    return captions


def cleaning_captions(captions):
    translation_table = str.maketrans("", "", string.punctuation)
    for image_id, list_of_captions in captions.items():
        for i in range(len(list_of_captions)):
            caption = list_of_captions[i]
            caption = caption.split()
            caption = [word.lower() for word in caption]
            caption = [word.translate(translation_table) for word in caption]
            caption = [word for word in caption if len(word) > 1]
            caption = [word for word in caption if word.isalpha()]
            list_of_captions[i] = " ".join(caption)


def transfer_captions_to_vocabulary(captions):
    all_captions = set()
    for image_id in captions.keys():
        [all_captions.update(d.split()) for d in captions[image_id]]
    return all_captions


def saving_ready_captions(captions, filename_to_save):
    rows = []
    for image_id, list_of_captions in captions.items():
        for caption in list_of_captions:
            rows.append(image_id + " " + caption)
    data = "\n".join(rows)
    with open(filename_to_save, "w") as saving_file:
        saving_file.write(data)


def prepare_text_data():
    print("Preparing text data was started! It may takes several minutes...")
    path_to_token = "Flickr8k_text/Flickr8k.token.txt"
    data = load_file(path_to_token)
    captions = extracting_captions(data)
    cleaning_captions(captions)
    vocabulary = transfer_captions_to_vocabulary(captions)
    saving_ready_captions(captions, "captions.txt")
    print("Preparing text data was finished!")