import pandas as pd
import numpy as np
from sklearn.manifold import Isomap, TSNE
import sklearn
import plotly.express as px
from sklearn.datasets import load_digits
import PySimpleGUI as sg
from dash import Dash, dcc, html

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
    [sg.Text('n_iter'), sg.InputText()],
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
    iters = int(values[2])

    iso = Isomap(n_neighbors = neighbors, n_components = dim)
    X_trans_iso = iso.fit_transform(X)

    tsne = TSNE(n_components = dim, perplexity = neighbors, n_iter = iters)
    X_trans_tsne = tsne.fit_transform(X)

    if dim == 3:
    # Create a 3D scatter plot
        fig_iso = px.scatter_3d(None, 
                            x=X_trans_iso[:,0], y=X_trans_iso[:,1], z=X_trans_iso[:,2],
                            color=y.astype(str),
                            height=600, width=600
                        )

        # Update chart looks
        fig_iso.update_layout(#title_text="Scatter 3D Plot",
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
        fig_iso.update_traces(marker=dict(size=5))

        fig_tsne = px.scatter_3d(None, 
                            x=X_trans_tsne[:,0], y=X_trans_tsne[:,1], z=X_trans_tsne[:,2],
                            color=y.astype(str),
                            height=600, width=600
                        )

        # Update chart looks
        fig_tsne.update_layout(#title_text="Scatter 3D Plot",
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
        fig_tsne.update_traces(marker=dict(size=5))

        # fig.show()
        # fig.write_html("tsne3.html")

    elif dim == 2:
        fig_iso = px.scatter(None, 
                            x=X_trans_iso[:,0], y=X_trans_iso[:,1],
                            color=y.astype(str),
                            height=600, width=600
                        )

        # Update chart looks
        fig_iso.update_layout(#title_text="Scatter 3D Plot",
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
        fig_iso.update_traces(marker=dict(size=5))

        # fig.show()
        # fig.write_html("isomap2.html")

        fig_tsne = px.scatter(None, 
                            x=X_trans_tsne[:,0], y=X_trans_tsne[:,1],
                            color=y.astype(str),
                            height=600, width=600
                        )

        # Update chart looks
        fig_tsne.update_layout(#title_text="Scatter 3D Plot",
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
        fig_tsne.update_traces(marker=dict(size=5))

        # fig.show()
        # fig.write_html("tsne3.html")

    else:
        print("Dimension can be only 2 or 3")

app = Dash(__name__)


app.layout = html.Div(className='row', children=[
    html.H1("Dimension Reduction"),
    # dcc.Dropdown(),
    html.Div(children=[
        html.H2("Isomap                                                             TSNE"),
        # html.H2("TSNE"),
        dcc.Graph(id="isomap", figure = fig_iso, style={'display': 'inline-block', 'width': '40%'}),
        dcc.Graph(id="tsne", figure = fig_tsne, style={'display': 'inline-block', 'width': '40%'})
    ])
])

if __name__ == '__main__':
    app.run_server()
