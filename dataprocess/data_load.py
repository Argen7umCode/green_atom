import os 


class DataImporter:
    
    def __init__(self, pathes) -> None:
        pathes = self.pathes

    def get_text_and_score(self):
        text_score = []
        for path in self.pathes:
            file_names = os.listdir(path)
            for file_name in file_names:
                with open(f'{path}/{file_name}', 'r') as file:
                    text = file.read()
                    score = int(file_name.split('.')[0].split('_')[-1])
                text_score.append([text, score])
        return text_score