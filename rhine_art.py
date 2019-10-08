#!/usr/bin/env python3

import argparse
import os
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# CLI Parser

parser = argparse.ArgumentParser('Generate templates for Rhinestone Artwork')
parser.add_argument('image_path', help='Path of source image')
parser.add_argument('save_directory', help='Directory to save the generated files')
parser.add_argument('-i', '--image-size', action='store', help='Image Size: Formats: 500x500 (width x height) or 500 (width only, preserve aspect ratio)')
parser.add_argument('-r', '--rhinestone-size', action='store', default='5x5', help='Rhine stone size. Format: 5x5')
parser.add_argument('-k', '--num-colors', action='store', type=int, default=10, help='Number of colors')
args = parser.parse_args()


# Validate arguments

image_path = args.image_path
if not os.path.isfile(image_path):
    raise Exception('File does not exist: {}'.format(image_path))
img_base_name = '.'.join(os.path.basename(image_path).split('.')[:-1])

save_dir = args.save_directory
if not os.path.isdir(save_dir):
    raise Exception('Directory does not exist: {}'.format(save_dir))

if args.image_size is None:
    target_size = None
else:
    matched = re.match('^(\d+)x(\d+)$', args.image_size)
    if matched is not None:
        target_size = tuple(map(int, matched.groups()))
        assert target_size[0] >= 1 and target_size[1] >= 1
    else:
        try:
            target_size = int(args.image_size), None
            assert target_size[0] >= 1
        except (ValueError, AssertionError):
            raise Exception('Invalid image size specified. Check usage.')

matched = re.match('^(\d+)x(\d+)$', args.rhinestone_size)
if matched is not None:
    stone_size = tuple(map(int, matched.groups()))
    try:
        assert stone_size[0] >= 1 and stone_size[1] >= 1
    except AssertionError:
        raise Exception('Invalid rhinestone size specified. Check usage.')
else:
    raise Exception('Invalid rhinestone size specified. Check usage.')

K = args.num_colors
if K < 2:
    raise Exception('At least 2 colors must be specified')


# Read image

img_orig = Image.open(image_path)


# Resize image

if target_size is None:
    target_size = img_orig.size[::-1]
    img_resized = img_orig
else:
    if target_size[1] is None:  # Preserving aspect ratio
        target_size[1] = round((target_size[0] * img_orig.size[0]) / img_orig.size[1])
    img_resized = img_orig.resize(target_size[::-1])


# Convert to Ndarray

img_resized_np = np.array(img_resized)


# Club pixels into blocks of stone size (This shrinks the image, allowing faster clustering)

img_stoned = np.zeros((
    int(np.ceil(img_resized_np.shape[0]/stone_size[0])),
    int(np.ceil(img_resized_np.shape[1]/stone_size[1])),
    img_resized_np.shape[2]
))

for i in range(img_stoned.shape[0]):
    for j in range(img_stoned.shape[1]):
        range_x = i*stone_size[0], min((i+1)*stone_size[0], img_resized_np.shape[0])
        range_y = j*stone_size[1], min((j+1)*stone_size[1], img_resized_np.shape[1])
        block = img_resized_np[range_x[0]:range_x[1], range_y[0]:range_y[1], :]
        block = block.reshape(-1, block.shape[2])
        img_stoned[i,j,:] = np.median(block, axis=0, keepdims=False)


# K-Means clustering to find K most representative colors

kmeans = KMeans(n_clusters=K).fit(img_stoned.reshape(-1, 3))


# Replace all the pixels in the image with the corresponding cluster centers

img_kmeans = np.zeros(img_stoned.shape).reshape(-1, 3)
for i in range(kmeans.labels_.size):
    img_kmeans[i, :] = kmeans.cluster_centers_[kmeans.labels_[i]]
img_kmeans = img_kmeans.reshape(img_stoned.shape)


# Convert image to original size

img_final = np.zeros_like(img_resized_np)

for i in range(img_kmeans.shape[0]):
    for j in range(img_kmeans.shape[1]):
        range_x = i*stone_size[0], min((i+1)*stone_size[0], img_resized_np.shape[0])
        range_y = j*stone_size[1], min((j+1)*stone_size[1], img_resized_np.shape[1])
        block = img_final[range_x[0]:range_x[1], range_y[0]:range_y[1], :]
        block[:,:,:] = img_kmeans[i,j,:]

# Save transformed image
img_name = '{}.rhinestone.png'.format(img_base_name)
img_path = os.path.join(save_dir, img_name)
Image.fromarray(img_final.astype(np.uint8)).save(img_path)
print('Transformed image saved to "{}"'.format(img_path))


# A utility method to get a contrasting color

def get_contrasting_color(r,g,b):
    return tuple(map(lambda x: 0 if x >= 128 else 255, (r, g, b)))


# A utility method to render text with contrasting color on an image block

def draw_text(draw, text, box_size, box_color):
    font = ImageFont.truetype('NotoSansMono-Light.ttf', int(min(box_size)/1.5))
    text_size = draw.textsize(text, font)
    text_center = ((box_size[0]-text_size[0])/2, (box_size[1]-text_size[1])/2)
    text_color = get_contrasting_color(*box_color)
    draw.text(text_center, text, fill=text_color, font=font)


# Generate color chart

rep_colors = []
rect_size = (200,200)

for i in range(K):
    color_rect = Image.new('RGB', rect_size)
    draw = ImageDraw.Draw(color_rect)
    rect_color = tuple(map(int, kmeans.cluster_centers_[i].tolist()))
    draw.rectangle(((0, 0), rect_size), fill=rect_color)
    draw_text(draw, str(i+1), rect_size, rect_color)
    rep_colors.append(np.array(color_rect))

# Generate a labeled color chart/grid
plots_per_row = max(2, int(np.ceil(np.sqrt(K))))
fig, axes = plt.subplots(plots_per_row, plots_per_row)
for i in range(plots_per_row**2):
    ax = axes[i // plots_per_row, i % plots_per_row]
    ax.axis('off')
    if i < K:
        ax.imshow(rep_colors[i])

# Save color chart
fig_name = '{}.colors.png'.format(img_base_name)
fig_path = os.path.join(save_dir, fig_name)
fig.savefig(fig_path)
print('Color chart saved to "{}"'.format(fig_path))


# Generate Rhinestone Artwork Template: Image with labels

img_labeled = np.zeros_like(img_final)
pix_labels = kmeans.labels_.reshape(img_kmeans.shape[0], img_kmeans.shape[1])

for i in range(pix_labels.shape[0]):
    for j in range(pix_labels.shape[1]):
        range_x = i*stone_size[0], min((i+1)*stone_size[0], img_labeled.shape[0])
        range_y = j*stone_size[1], min((j+1)*stone_size[1], img_labeled.shape[1])
        block_labelled = img_labeled[range_x[0]:range_x[1], range_y[0]:range_y[1], :]
        block_final = img_final[range_x[0]:range_x[1], range_y[0]:range_y[1], :]
        
        label = pix_labels[i,j]
        img_block = Image.fromarray(block_final.astype(np.uint8))
        draw = ImageDraw.Draw(img_block)
        block_color = tuple(map(int, kmeans.cluster_centers_[label].tolist()))
        draw_text(draw, str(label+1), block_final.shape[:-1], block_color)
        block_labelled[:,:,:] = np.array(img_block)

# Save template image
img_name = '{}.rhinestone.template.png'.format(img_base_name)
img_path = os.path.join(save_dir, img_name)
Image.fromarray(img_labeled.astype(np.uint8)).save(img_path)
print('Template image saved to "{}"'.format(img_path))