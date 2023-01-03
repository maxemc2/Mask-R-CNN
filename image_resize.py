import os
from PIL import Image

def resize(path, type, size):
  imgs = list(sorted(os.listdir(os.path.join("class_data", path, type, "image"))))
  masks = list(sorted(os.listdir(os.path.join("class_data", path, type, "mask"))))
  new_img_dir = os.path.join("class_data", path, type, "resized_image")
  new_mask_dir = os.path.join("class_data", path, type, "resized_mask")

  if not os.path.isdir(new_img_dir):
    os.mkdir(new_img_dir)
  if not os.path.isdir(new_mask_dir):
    os.mkdir(new_mask_dir)

  for index in range(len(imgs)):
    img_path = os.path.join("class_data", path, type, "image", imgs[index])
    img = Image.open(img_path)
    img = img.resize((int(img.size[0] * size), int(img.size[1] * size)), resample=Image.NEAREST) 
    img.save(os.path.join("class_data", path, type, "resized_image", imgs[index]))
    mask_path = os.path.join("class_data", path, type, "mask", masks[index])
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((int(mask.size[0] * size), int(mask.size[1] * size)), resample=Image.NEAREST) 
    mask.save(os.path.join("class_data", path, type, "resized_mask", masks[index]))

#test size: 1/6, 1/2 and 1/2, 1
resize("Train", "powder_uncover", 1/6)
resize("Train", "powder_uneven", 1/6)
resize("Train", "scratch", 1/2)
print("Done!")