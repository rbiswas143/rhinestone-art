# rhinestone-art
A [Rinestone](https://en.wikipedia.org/wiki/Rhinestone) is a tiny diamond like crystal or stone. Rhinestone Art is like cross strich but with these stones. The repository includes a Python script that allows you to convert your favorite image to a template for creating Rhinestone Artwork. You can specifiy the size of the image and stones, and the number of colors needed, and the script will generate a color chart that will allow you to procure the required stones and a template image that tells you where exactly to place each stone

Internally, the code uses (K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering) to generate the template. Go through [walkthrough.ipynb](https://github.com/rbiswas143/rhinestone-art/blob/master/walkthrough.ipynb) to understand the code

I originally wrote this for my artistic sister, but here you all go!

### Installation
Install [Python 3](https://www.python.org/download/releases/3.0/) and all other dependencies in ```deps.py``` using pip or your favorite package manager. Place the script in any location in your path or simply invoke with Python.

### Usage
```python rhinestone_art.py <path_to_source_image> <destination_directory>```

You can specify the output tempalte size in pixels<br>
```python rhinestone_art.py -i 400x600 lion.jpg ./```

Or simply the template width, to preserve aspect ratio<br>
```python rhinestone_art.py -i 400 lion.jpg ./```

You can additionally specify the stone size<br>
```python rhinestone_art.py -r 10x10 lion.jpg ./```

And the number of colors in the final template<br>
```python rhinestone_art.py -k 15 lion.jpg ./```

### Results

**Source Image**

![Original Image](https://github.com/rbiswas143/rhinestone-art/blob/master/images/lion/lion.jpg)

**Template Image**

![Template Image](https://github.com/rbiswas143/rhinestone-art/blob/master/images/lion/lion.rhinestone.template.png)

**Color Chart**

![Color Chart](https://github.com/rbiswas143/rhinestone-art/blob/master/images/lion/lion.colors.png)
