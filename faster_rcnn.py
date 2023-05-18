import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection
from torchvision import transforms
from torchvision.ops import box_iou
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_tensor

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)

transform = ToTensor()
trainval_data = VOCDetection(root=".", image_set="trainval", download=True, transforms=transform)


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def preprocess_targets(targets):
    new_targets = []
    for target in targets:
        boxes = []
        labels = []
        for obj in target["annotation"]["object"]:
            bbox = obj["bndbox"]
            box = [int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])]
            boxes.append(box)
            labels.append(VOC_CLASSES.index(obj["name"]))
        new_targets.append({"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)})
    return new_targets

def calculate_image_precision(pred_boxes, true_boxes, thresholds = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75)):
    precisions = []
    for threshold in thresholds:
        fp = tp = 0
        for pred_box in pred_boxes:
            ious = box_iou(true_boxes, pred_box[0].unsqueeze(0))
            if torch.any(ious >= threshold):
                tp += 1
            else:
                fp += 1
        if (tp+fp) == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        precisions.append(precision)
    return 1 - sum(precisions) / len(precisions)
    

train_size = int(0.8 * len(trainval_data))
val_size = len(trainval_data) - train_size
train_data, val_data = random_split(trainval_data, [train_size, val_size])

test_loader = torch.utils.data.DataLoader(
    val_data, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=lambda x: tuple(zip(*x)))


model.eval()
with torch.no_grad():
    results = []
    image_precisions = []
    for images, targets in test_loader:
        images = [image.to(device) for image in images]
        targets = preprocess_targets(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        output = model(images)

        preds = []
        for box, score in zip(output[0]['boxes'], output[0]['scores']):
            preds.append((box, score))
        image_precision = calculate_image_precision(preds, targets[0]['boxes'])
        image_precisions.append(image_precision)

    mAP = sum(image_precisions) / len(image_precisions)
    print(f'mAP: {mAP}')


num_classes = 21
# Get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)


def train_model(model, train_loader, optimizer, num_epochs=2):
    model.train()
    for epoch in range(num_epochs):
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = preprocess_targets(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

def collate_fn(batch):
    try:
        return tuple(zip(*batch))
    except Exception as e:
        print(f"Exception while processing batch: {e}")
        for item in batch:
            try:
                # Attempt to process each item in the batch individually
                tuple(zip(*item))
            except Exception as e:
                print(f"Exception for item {item}: {e}")
        # Reraise the original exception
        raise e

# train_loader = torch.utils.data.DataLoader(
#     train_data, batch_size=2, shuffle=True, num_workers=4,
#     collate_fn=lambda x: tuple(zip(*x)))

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=collate_fn)


# Train the model
train_model(model, train_loader, optimizer, num_epochs=2)

# After training, save the model for future use or evaluation
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn_finetuned.pth')

# Now evaluate the model with the evaluation code you have already written
model.eval()
with torch.no_grad():
    # Run the model on the test data and gather the results
    results = []
    image_precisions = []
    for images, targets in test_loader:
        images = [image.to(device) for image in images]
        targets = preprocess_targets(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        output = model(images)

        preds = []
        for box, score in zip(output[0]['boxes'], output[0]['scores']):
            preds.append((box, score))
        image_precision = calculate_image_precision(preds, targets[0]['boxes'])
        image_precisions.append(image_precision)

    # Compute mean Average Precision (mAP)
    mAP = sum(image_precisions) / len(image_precisions)
    print(f'mAP: {mAP}')