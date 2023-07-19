import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import offline
import numpy as np


class DataCharts:

    def __init__(self) -> None:
        super().__init__()
        self.x = None
        self.y = None
        self.z = None
        self.labels = None

    def GetCoordinate(self, data, labels):
        self.x = []
        self.y = []
        self.z = []
        self.labels = []

        for point in data:
            self.x.append(point[0])
            self.y.append(point[1])
            self.z.append(point[2])
            if labels[data.index(point)] == 0:
                self.labels.append('#5470c6')
            else:
                self.labels.append('#91cc75')

    def ScatterPlot3D(self, coordinate, category):
        # print(self.x)
        self.GetCoordinate(coordinate, category)
        # print(self.x)

        data = [go.Scatter3d(x=self.x, y=self.y, z=self.z, mode='markers', marker=dict(size=12, color=self.labels))]

        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(nticks=4),
                yaxis=dict(nticks=4),
                zaxis=dict(nticks=4)
            )
        )
        fig = go.Figure(data=data, layout=layout)
        offline.plot(fig, filename='static/Pictures/scatter.html', auto_open=False)

        return 0

    def ScatterPlot2D(self, x, y, labels):
        self.x = x
        self.y = y
        self.labels = labels
        return 0

    def GetCoordinate_1(self, x_test, y_test, y_pred):
        self.x = []
        self.y = y_test
        self.z = y_pred
        i = 0
        for point in x_test:
            self.x.append(i)
            i += 1

    def LineChart(self, x, y_true, y_pred):
        self.GetCoordinate_1(x, y_true, y_pred)

        data = [go.Scatter(
            x=self.x, y=self.y,
            mode='markers+lines',
            marker=dict(symbol='circle',
                        size=10,
                        color='#FB8D75'),  # More options here
            name='true value'
        ),
            go.Scatter(
                x=self.x, y=self.z,
                mode='markers+lines',
                marker=dict(symbol='circle',
                            size=10,
                            color='#659B91'),  # More options here
                name='predicted value'
            )
        ]

        layout = go.Layout(title='2D Line Graph')

        fig = go.Figure(data=data, layout=layout)

        offline.plot(fig, filename='static/Pictures/scatter.html', auto_open=False)


if __name__ == '__main__':
    print("begin...")
    data = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
    true_labels = [1, 2, 3, 4]
    pred_labels = [4, 3, 2, 1]
    labels = [1, 1, 0, 0]
    pic = DataCharts()
    # pic.ScatterPlot3D(data,labels)
    pic.LineChart(data, true_labels, pred_labels)
