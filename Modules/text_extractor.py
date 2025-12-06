import json
import re


def data_extractor(filename):
    results = []
    with open(filename, 'r', encoding = 'utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            data = json.loads(line)
            results.append(data)                
    return results


def news_extractor(filename, category = 'all'):
    results = []
    with open(filename, 'r', encoding = 'utf-8') as file:
        if (category == 'all'):
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                data = json.loads(line)
                results.append(data['text'])
        else:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                data = json.loads(line)
                if (data['category'] == category):
                    results.append(data['text'])                  
    return results


def news_to_text_converter(news):
    text = ' '.join(news)
    return text


def text_extractor(filename, category = 'all'):
    news = news_extractor(filename, category)
    text = news_to_text_converter(news)
    return text


def vectors_extractor(filename, category = 'all'):
    results = []
    with open(filename, 'r', encoding = 'utf-8') as file:
        if (category == 'all'):
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                data = json.loads(line)
                results.append(data['text'])
        else:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                data = json.loads(line)
                if (data['category'] == category):
                    results.append(data['text'])
    for r in range(len(results)):
        results[r] = [float(x) for x in results[r].split(' ')]
    return results


