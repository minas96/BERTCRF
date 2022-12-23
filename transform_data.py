import sys
import os
import nltk
from nltk.tokenize import word_tokenize
import json

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)

    def get_path(dirname, type, file_type =0):
        url = "data\\" + type
        # file_type = 0 -> .txt
        # file_type = 1 -> .ann
        if file_type == 0:
            url = url + "\\text\\"
        else:
            url = url + "\\attributions\\"
        return  os.path.join(dirname, url)

    type_of_dataset = sys.argv[1]
    print("Transform ",type_of_dataset," data")
    articles_names = os.listdir(get_path(dirname, type_of_dataset))
    attributions_names = os.listdir(get_path(dirname, type_of_dataset, 1))

    def get_attribution_filename_by_article_name(article_name, attributions_names):
        for attribution in attributions_names:
            if article_name[:-4] in attribution:
                return attribution

    def get_article(dirname, type, article_name, skip_line=0):
        path = get_path(dirname, type)
        f = open(path+article_name, "r", encoding="utf8")
        article = []
        for x in f:
            #Remove last characther "\n" if exists
            if(x[len(x)-1]=="\n"):
                x = x[:len(x)-1]
            article.append(x)
            for i in range(skip_line):
                f.readline()
        f.close()
        return article

    articles = {}
    print("Read articles")
    for articles_name in articles_names:
        articles[articles_name] = get_article(dirname, type_of_dataset, articles_name, 1)

    #use brat parser to decode .ann files
    import brat_parser
    from brat_parser import get_entities_relations_attributes_groups, Entity

    # it is used in quicksort
    def partition(array, low, high):
        pivot = array[high]
        i = low - 1
        for j in range(low, high):
            start_item =  array[j].span[0][0] if type(array[j].span[0]) is tuple else array[j].span[0]
            start_pivot = pivot.span[0][0] if type(pivot.span[0]) is tuple else pivot.span[0]
            if start_item <= start_pivot:
                i = i + 1
                (array[i], array[j]) = (array[j], array[i])
        (array[i + 1], array[high]) = (array[high], array[i + 1])

        return i + 1

    # https://www.geeksforgeeks.org/python-program-for-quicksort/
    def quickSort(array, low, high):
        if low < high:
            pi = partition(array, low, high)
            quickSort(array, low, pi - 1)
            quickSort(array, pi + 1, high)

    def get_entities_from_ann(dirname, type, filename):
        file = get_path(dirname, type, 1) + filename
        entities, relations, attributes, groups = get_entities_relations_attributes_groups(file)
        desired_entitites = []
        for id in entities:
            if(entities[id].type == "Cue" or entities[id].type == "Source" or entities[id].type == "Content"):
                start_index = 0
                end_index = 0
                i = 0
                # split into multiple entities if entity has discontinuous span
                if len(entities[id].span) > 1:
                    for span in entities[id].span:
                        end_index = end_index + span[1] - span[0] + 1
                        entity = Entity(id = str(id)+"_"+str(i), type = entities[id].type, span = (span[0], span[1]), text = entities[id].text[start_index: end_index])
                        desired_entitites.append(entity)
                        start_index = end_index
                        i = i + 1
                else:
                    desired_entitites.append(entities[id])

        size = len(desired_entitites)
        # sort entities in order to have ascending spans
        quickSort(desired_entitites, 0, size - 1)
        return desired_entitites

    attributions = {}
    print("Read attributions")
    for attributions_name in attributions_names:
        attributions[attributions_name] = get_entities_from_ann(dirname, type_of_dataset, attributions_name)

    # def get_percentage_of_quotations_in_article(article, attributions):
    #     start = 0
    #     counter = 0
    #     for instance in article:
    #         for index in range(len(attributions)):
    #             print(attributions[index]," ", attributions[index].span)
    #             if attributions[index].start >= start and attributions[index].end <= (start+len(instance)-1):
    #                 counter = counter + 1
    #                 break
    #         start = start + len(instance)
    #     return counter / len(article)

    # per = get_percentage_of_quotations_in_article(articles[articles_names[0]],attributions[attribution_filename])
    # print("Number of instances: ", len(articles[articles_names[0]]), " / Percentage of instances that contain source, cue or content: ", round(per*100,2), "%")

    class Article:
        def __init__(self):
            self.article_name = ""
            self.attribution_name = ""
            self.article = []
            self.attributions = []

    articles_items = []
    for article_name in articles_names:
        article = Article()
        article.article_name = article_name
        article.attribution_name = get_attribution_filename_by_article_name(article_name, attributions_names)
        article.article = articles[article_name]
        article.attributions = attributions[article.attribution_name]
        articles_items.append(article)

    final_data = []
    print("Construct final data to store in file")
    for article_item in articles_items:
        instances = {}
        start = 0
        for instance in article_item.article:
            instances[instance] = (start, start + len(instance))
            start = start + len(instance) + 2

        instances_keys = list(instances.keys())

        def reformat_data(splitted_instance, splitted_instance_tags):
            final_splitted_instance = []
            final_splitted_instance_tags = []
            for j in range(len(splitted_instance)):
                splitted_instance_item = splitted_instance[j]
                splitted_instance_tags_item = splitted_instance_tags[j]
                excepted_symbols = []#[']', '_', '{', ':', '(', '}', '$', ';', ')', '[', '%', '#', '@', '-', '^', '`', '\\', '?', '|', ',', '/', '~', '>', '<', '!', '=', '.', '+', '*', '&']
                temp_tokens = word_tokenize(splitted_instance_item)
                tokens=[]
                for token in temp_tokens:
                    if token not in excepted_symbols:
                        tokens.append(token)
                final_splitted_instance.extend(tokens)
                for i in range(len(tokens)):
                    if splitted_instance_tags_item == 'Source' and i == 0:
                        final_splitted_instance_tags.append('B-Source')
                    elif splitted_instance_tags_item == 'Source':
                        final_splitted_instance_tags.append('I-Source')
                    elif splitted_instance_tags_item == 'Cue' and i == 0:
                        final_splitted_instance_tags.append('B-Cue')
                    elif splitted_instance_tags_item == 'Cue':
                        final_splitted_instance_tags.append('I-Cue')
                    elif splitted_instance_tags_item == 'Content' and i == 0:
                        final_splitted_instance_tags.append('B-Content')
                    elif splitted_instance_tags_item == 'Content':
                        final_splitted_instance_tags.append('I-Content')
                    else:
                        final_splitted_instance_tags.append('O')
            return final_splitted_instance, final_splitted_instance_tags

        tagged_instances = []
        for key in instances_keys:
            splitted_instance = [key]
            splitted_instance_tags = ['O']
            temp_index = instances[key][0]
            for attribution in articles_items[0].attributions:
                    attribution_start = attribution.span[0][0] if type(attribution.span[0]) is tuple else attribution.span[0]
                    attribution_end = attribution.span[0][1] if type(attribution.span[0]) is tuple else attribution.span[1]
                    if instances[key][0] <= attribution_start and attribution_end <= instances[key][1]:
                        last_item = splitted_instance.pop()
                        splitted_instance_tags.pop()
                        if last_item[:attribution_start - temp_index] != "" and last_item[:attribution_start - temp_index] != " " :
                            splitted_instance.append(last_item[:attribution_start-temp_index])
                            splitted_instance_tags.append('O')
                        splitted_instance.append(last_item[attribution_start - temp_index:attribution_end - temp_index])
                        splitted_instance_tags.append(attribution.type)
                        if last_item[attribution_end- temp_index:] != "" and last_item[attribution_end - temp_index:] != " ":
                            splitted_instance.append(last_item[attribution_end -temp_index:])
                            splitted_instance_tags.append('O')
                        temp_index = attribution_end
                    if instances[key][1] < attribution_start:
                        break

            splitted_instance, splitted_instance_tags = reformat_data(splitted_instance, splitted_instance_tags)
            tagged_instances.append([splitted_instance, splitted_instance_tags])
        final_data.append(tagged_instances)

    json_string = json.dumps(final_data)
    f = open(os.path.join(dirname, type_of_dataset+"_data.json"), "a")
    f.write(json_string)
    f.close()