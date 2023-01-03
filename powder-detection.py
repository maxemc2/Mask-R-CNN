import os
import numpy as np
from PIL import Image
import torch, torchvision
import torch.utils.data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import transforms as T
from engine import train_one_epoch, evaluate

#three type of defect and background
CLASS_NUM = 4
EPOCHS_NUM = 10
BATCH_SIZE = 2
NUM_WORKER = 4

def getAllFilePaths(directory):
  for dirpath,_,filenames in os.walk(directory):
      for f in filenames:
          yield os.path.abspath(os.path.join(dirpath, f))

class PowderDataset(torch.utils.data.Dataset):
  def __init__(self, root, transforms=None):
    self.root = root
    self.transforms = transforms
    self.image_paths = []
    self.mask_paths = []
    # there are three types of defect: uncovered powder, uneven powder and scratched powder,
    # store all training file absolute paths including images and masks, and sort them to ensure that they are aligned    
    defect_type = ["powder_uncover", "powder_uneven", "scratch"]
    for type_path in defect_type:
      self.image_paths += list(sorted(file for file in getAllFilePaths(os.path.join(root, type_path, "resized_image"))))
      self.mask_paths += list(sorted(file for file in getAllFilePaths(os.path.join(root, type_path, "resized_mask"))))

  def __getitem__(self, index):
    # load images and masks
    img_path = self.image_paths[index]
    mask_path = self.mask_paths[index]
    # note that we haven't converted the mask to RGB,
    # because each color corresponds to a different instance with 0 being background
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)

    mask = np.array(mask)
    # get all type of color, and first color is background, so remove it
    obj_colors = np.unique(mask)[1:]
    # split the color-encoded mask into a set of binary masks
    masks = mask == obj_colors[:, None, None]
    
    # get bounding box coordinates for each mask
    obj_num = len(obj_colors)
    boxes = []
    for i in range(obj_num):
      pos = np.where(masks[i])
      xmin = np.min(pos[1])
      xmax = np.max(pos[1])
      ymin = np.min(pos[0])
      ymax = np.max(pos[0])
      boxes.append([xmin, ymin, xmax, ymax])

    boxes = torch.as_tensor(boxes, dtype=torch.float32)

    if "powder_uncover" in mask_path:
      labels = torch.ones(obj_num, dtype=torch.int64)
    elif "powder_uneven" in mask_path:
      labels = torch.as_tensor([2 for i in range(obj_num)], dtype=torch.int64)
    else: #"scratch" in mask_path
      labels = torch.as_tensor([3 for i in range(obj_num)], dtype=torch.int64)
      
    masks = torch.as_tensor(masks, dtype=torch.uint8)
    image_id = torch.tensor([index])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # suppose all instances are not crowd
    iscrowd = torch.zeros((obj_num,), dtype=torch.int64)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["masks"] = masks
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    if self.transforms is not None:
      img, target = self.transforms(img, target)

    return img, target

  def __len__(self):
      return len(self.image_paths)

def get_instance_segmentation_model(num_classes):
  #load an instance segmentation model pre-trained on COCO
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(True)
  #get the number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  # now get the number of input features for the mask classifier
  in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
  hidden_layer = 256
  # and replace the mask predictor with a new one
  model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

  return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

def train_model():
  dataset = PowderDataset("PowderDataset", get_transform(train=True))
  test_dataset = PowderDataset("PowderDataset", get_transform(train=False))
  # randomly arrange training order
  torch.manual_seed(1)
  indices = torch.randperm(len(dataset)).tolist()
  # split in 3:1 according 4 fold cross validation
  dataset = torch.utils.data.Subset(dataset, indices[:-112])
  test_dataset = torch.utils.data.Subset(test_dataset, indices[-112:])

  #define training and validation data loaders
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER,
                                            collate_fn=utils.collate_fn)

  data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER,
                                                collate_fn=utils.collate_fn)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  # get the model using the helper function
  # if os.path.isfile('powder-parameter.pth'):
  #   model = torch.load('powder-parameter.pth')
  model = get_instance_segmentation_model(CLASS_NUM)
  # move model to the right device
  model.to(device)

  # construct an optimizer
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.005,
                              momentum=0.9, weight_decay=0.0005)
  # the learning rate scheduler decreases the learning rate by 10x every 3 epochs
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

  # training
  for epoch in range(EPOCHS_NUM):
      # train for one epoch, printing every 10 iterations
      train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
      # update the learning rate
      lr_scheduler.step()
      # evaluate on the test dataset
      evaluate(model, data_loader_test, device=device)

  torch.save(model, 'powder-parameter.pth')

def test_model():
  test_dataset = PowderDataset("PowderDataset", get_transform(train=False))
  
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = torch.load('powder-parameter.pth')

  img, _ = test_dataset[0]
  # put the model in evaluation mode
  model.eval()
  with torch.no_grad():
    prediction = model([img.to(device)])

  Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).show()
  Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()).show()

if __name__ == '__main__':
  train_model()
