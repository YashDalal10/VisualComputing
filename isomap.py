import pandas as pd
import numpy as np
from sklearn.manifold import Isomap
import sklearn
import plotly.express as px
from sklearn.datasets import load_digits
import PySimpleGUI as sg

X,y = load_digits(return_X_y = True)
X.shape, y.shape

layout = [
    [sg.Text('Number of neighbors (default=5)'), sg.InputText()],
    [sg.Text('radius (default=None) (helpful for limiting the distance')],
    [sg.Text('Number of dimensions (can take only 2 or 3'), sg.InputText()],
    [sg.Text('eigen_solver (default=auto) (arpack - Arnoldi decomposition or dense - LAPACK)')],
    [sg.Text('Path method (default=auto)(FW - Floyd-Warshall or D - Dijkstras)')],
    [sg.Text('Neighbors algorithm (default=auto)(brute, kd tree and ball tree)')],
    [sg.Text('n_jobs (default=-1, all processors)')],
    [sg.Button('Reduce and Plot')]
]
# neighbors = int(input("The no of neighbors: "))
# components = int(input("The no of dimensions to be reduced into: "))
window = sg.Window('Parameters', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    print('Nearest Neighbors: ', values[0])
    print('Dimension to reduce: ', values[1])
    neighbors = int(values[0])
    dim = int(values[1])

    iso = Isomap(n_neighbors = neighbors, n_components = dim)
    X_trans = iso.fit_transform(X)

    if dim == 3:
    # Create a 3D scatter plot
        fig = px.scatter_3d(None, 
                            x=X_trans[:,0], y=X_trans[:,1], z=X_trans[:,2],
                            color=y.astype(str),
                            height=900, width=900
                        )

        # Update chart looks
        fig.update_layout(#title_text="Scatter 3D Plot",
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                        scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                                center=dict(x=0, y=0, z=-0.2),
                                                eye=dict(x=-1.5, y=1.5, z=0.5)),
                                                margin=dict(l=0, r=0, b=0, t=0),
                        scene = dict(xaxis=dict(backgroundcolor='white',
                                                color='black',
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                ),
                                    yaxis=dict(backgroundcolor='white',
                                                color='black',
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                ),
                                    zaxis=dict(backgroundcolor='lightgrey',
                                                color='black', 
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                )))

        # Update marker size
        fig.update_traces(marker=dict(size=5))

        fig.show()
        fig.write_html("plotly1.html")

    elif dim == 2:
        fig = px.scatter(None, 
                            x=X_trans[:,0], y=X_trans[:,1],
                            color=y.astype(str),
                            height=900, width=900
                        )

        # Update chart looks
        fig.update_layout(#title_text="Scatter 3D Plot",
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                        scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                                center=dict(x=0, y=0, z=-0.2),
                                                eye=dict(x=-1.5, y=1.5, z=0.5)),
                                                margin=dict(l=0, r=0, b=0, t=0),
                        scene = dict(xaxis=dict(backgroundcolor='white',
                                                color='black',
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                ),
                                    yaxis=dict(backgroundcolor='white',
                                                color='black',
                                                gridcolor='#f0f0f0',
                                                title_font=dict(size=10),
                                                tickfont=dict(size=10),
                                                )))

        # Update marker size
        fig.update_traces(marker=dict(size=5))

        fig.show()
        fig.write_html("plotly1.html")

    else:
        print("Dimension can be only 2 or 3")
