from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random
import pytesseract


def add_speckle_noise(image, sigma=0.2):
    data = np.array(image)
    h, w = data.shape
    noise = np.random.randn(h, w)
    noisy_image = data + data * noise * sigma
    return Image.fromarray(np.clip(noisy_image, 0, 255).astype(np.uint8))

def random_zoom(image, zoom_range=(0.9, 1.1)):
    # Get the original image size
    width, height = image.size

    # Calculate the center of the image
    center_x = width / 2
    center_y = height / 2

    zoom_factor = random.uniform(zoom_range[0], zoom_range[1])

    # Calculate the new size after applying the zoom factor
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    # Calculate the new top-left corner to keep the image centered
    left = int(center_x - new_width / 2)
    top = int(center_y - new_height / 2)

    # Calculate the new bottom-right corner
    right = left + new_width
    bottom = top + new_height
    zoomed_image = image.crop((left, top, right, bottom))
    zoomed_image = zoomed_image.resize((width, height), Image.ANTIALIAS)

    return zoomed_image

def random_occlusion_with_blur(image, occlusion_size, blur_radius):
    occluded_image = image.copy()
    image_width, image_height = occluded_image.size

    # random crop
    left = random.randint(0, image_width - occlusion_size[0])
    top = random.randint(0, image_height - occlusion_size[1])
    right = left + occlusion_size[0]
    bottom = top + occlusion_size[1]
    occlusion_coords = (left, top, right, bottom)
    occlusion_region = occluded_image.crop(occlusion_coords)


    occlusion_region = occlusion_region.filter(ImageFilter.GaussianBlur(blur_radius))
    occluded_image.paste(occlusion_region, occlusion_coords)

    return occluded_image

def random_rotation_translation(image, max_angle=5, max_shift=(5, 5)):
    angle = random.uniform(-max_angle, max_angle)
    img = image.rotate(angle, resample=Image.BICUBIC, expand=True)

    shift_x = random.randint(-max_shift[0], max_shift[0])
    shift_y = random.randint(-max_shift[1], max_shift[1])

    img = img.transform(image.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y), resample=Image.BICUBIC)
    return img

def copy_move(img):
    w, h = img.size
    # Perform OCR using pytesseract to get the bounding boxes
    boxes = pytesseract.image_to_boxes(img)
    i = 0

    if len(boxes.splitlines()) == 0:
       roll = random.randint(0,3)
       if roll == 0:
          img = add_speckle_noise(img,sigma =0.2)
       elif roll == 1:
          img = random_zoom(img)
       elif roll == 2:
          img = random_occlusion_with_blur(img, occlusion_size=(100, 100), blur_radius=10)
       else:
          img = random_rotation_translation(img)
       return img
    else:
      end = random.randint(1, len(boxes.splitlines()))
      first, second, third, fourth = None, None, None, None

      for b in boxes.splitlines():
          b = b.split(' ')
          i += 1
          if i == end:
              # Draw a green rectangle around the selected bounding box
              x1, y1, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
              img_with_rectangle = img.copy()  # Make a copy of the image to draw on it
              draw = ImageDraw.Draw(img_with_rectangle)
              draw.rectangle([x1, y1, x2, y2], outline=255, width=2)  # Color specified as an integer
              first, second, third, fourth = x1, y1, x2, y2
              break
      x = random.randint(0,w)
      y = random.randint(0,h)
      cropped = img.crop((first,fourth,third,second))
      img.paste(cropped,(x,y))
      return img

def full_transformations(image):
    # add speckle noise
    number = random.randint(0,3)
    if number == 0:
      img = add_speckle_noise(image,sigma =0.2)
    elif number == 1:
      img = random_zoom(image)
    elif number == 2:
      img = random_occlusion_with_blur(image, occlusion_size=(100, 100), blur_radius=10)
    elif number == 3:
      img = random_rotation_translation(image)
    else:
      img = copy_move(image) 

    return img

