import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

df = pd.read_csv("data/winemag-data-130k-v2.csv", index_col=0)

text = " ".join(review for review in df.description)
print("There are {} words in the combination of all review.".format(len(text)))

stopwords = set(STOPWORDS)
stopwords.update("drink", "now", "wine", "made", "the")

swarna = np.array(Image.open("img/Sw1f.jpg"))
wc = WordCloud(background_color="white", mask=swarna,
               contour_width=1, contour_color='firebrick')
wc.generate(text)
image_colors = ImageColorGenerator(swarna)
wc.recolor(color_func=image_colors)
wc.to_file("img/sw.png")

plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
