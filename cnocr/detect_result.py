class detect_result:
    def __init__(self, result, class_ids_length, end_idx):
        self.result = result
        self.class_ids_length = class_ids_length
        self.end_idx = end_idx
    
    @staticmethod
    def create(result, class_ids_length, end_idx):
        result = result
        class_ids_length = class_ids_length
        end_idx = end_idx
        return detect_result(result, class_ids_length, end_idx)