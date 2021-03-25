import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['10', '5', '3', '1']
men_means = [45.19, 39.92, 35.81, 23.47]
women_means = [55.07, 51.86, 46.64, 29.29]
other_means = [55.87, 52.51, 49.00, 33.30]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 0.25, men_means, width, label='Entire Query', color=(0.3, 0.5, 0.7, 0.9))
rects2 = ax.bar(x , women_means, width, label='Relevant Tokens', color=(0.2, 0.7, 0.9, 0.9))
rects3 = ax.bar(x + 0.25, other_means, width, label='Relevant Tokens + Context', color=(0.2, 0.9, 0.7, 0.5))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy Score')
ax.set_title('BM25 Accuracy Scores')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{} %'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 3, height),
                    xytext=(0, -1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()