import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model():
    # Load a pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Number of classes : 0 for background + 1 for sunspot
    num_classes = 2

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace original classification head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
