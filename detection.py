

class detections:

    def detection(self, bounding_box,confidence_score, class_name):
        self.bounding_box = bounding_box
        self.confidence_score = confidence_score
        self.class_name = class_name