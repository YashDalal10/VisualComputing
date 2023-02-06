import pandas as pd
import numpy as np
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
import sklearn
import plotly.express as px
from sklearn.datasets import load_digits
import PySimpleGUI as sg
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

X,y = load_digits(return_X_y = True)
X.shape, y.shape

input_file = [[sg.Text("Choose a file: "), sg.Input(), ]]

layout_algo = [
    [sg.Text('Isomap algorithm requires follwoing parameters - ')],
    [sg.Text('Number of neighbors (default=5)'), sg.InputText()],
    [sg.Text('Number of dimensions (can take only 2 or 3)'), sg.InputText()],
    [sg.Text('TSNE algorithm requires follwoing parameters - ')],
    [sg.Text('Perplexity (no of neighbors) (default=30.0)'), sg.InputText()],
    [sg.Text('Number of dimensions (can take only 2 or 3)'), sg.InputText()],
    [sg.Text('Number of iterations (default 1000)'), sg.InputText()],
    [sg.Text('PCA algorithm requires follwoing parameters - ')],
    [sg.Text('Number of dimensions (can take only 2 or 3)'), sg.InputText()],
    [sg.Button('Reduce and Plot')]
]

window_algo = sg.Window('Parameters', layout_algo)
while True:
    event_algo, values_algo = window_algo.read()
    if event_algo == sg.WIN_CLOSED:
        break
    print('Isomap Nearest Neighbors: ', values_algo[0])
    print('Isomap no of dimensions to reduce to: ', values_algo[1])
    print('TSNE Nearest Neighbors: ', values_algo[2])
    print('TSNE no of dimensions to reduce to: ', values_algo[3])
    print('TSNE no of iterations: ', values_algo[4])
    print('PCA no of dimensions to reduce to: ', values_algo[5])
    neighbors_iso = int(values_algo[0])
    dim_iso = int(values_algo[1])
    neighbors_tsne = int(values_algo[2])
    dim_tsne = int(values_algo[3])
    iter_tsne = int(values_algo[4])
    dim_pca = int(values_algo[5])


    if dim_iso != dim_tsne or dim_iso != dim_pca:
        print('Error')

    iso = Isomap(n_neighbors = neighbors_iso, n_components = dim_iso)
    X_trans_iso = iso.fit_transform(X)

    tsne = TSNE(n_components = dim_tsne, perplexity = neighbors_tsne, n_iter = iter_tsne)
    X_trans_tsne = tsne.fit_transform(X)

    pca = PCA(n_components = dim_pca)
    X_trans_pca = pca.fit_transform(X)

    if dim_iso == 3 and dim_tsne == 3 and dim_pca == 3:
    # Create a 3D scatter plot
        fig_iso = px.scatter_3d(None, 
                            x=X_trans_iso[:,0], y=X_trans_iso[:,1], z=X_trans_iso[:,2],
                            color=y.astype(str),
                            height=700, width=700, title = 'Isomap 3D'
                        )

        fig_iso.update_traces(marker=dict(size=3))

        fig_tsne = px.scatter_3d(None, 
                            x=X_trans_tsne[:,0], y=X_trans_tsne[:,1], z=X_trans_tsne[:,2],
                            color=y.astype(str),
                            height=700, width=700, title = 'TSNE 3D'
                        )

        fig_tsne.update_traces(marker=dict(size=3))

        fig_pca = px.scatter_3d(None, 
                            x=X_trans_pca[:,0], y=X_trans_pca[:,1], z=X_trans_pca[:,2],
                            color=y.astype(str),
                            height=700, width=700, title = 'PCA 3D'
                        )

        fig_tsne.update_traces(marker=dict(size=3))


    elif dim_iso == 2 and dim_tsne == 2 and dim_pca == 2:
        fig_iso = px.scatter(None, 
                            x=X_trans_iso[:,0], y=X_trans_iso[:,1],
                            color=y.astype(str),
                            height=600, width=600, title = 'Isomap 2D'
                        )

        
        fig_iso.update_traces(marker=dict(size=5))


        fig_tsne = px.scatter(None, 
                            x=X_trans_tsne[:,0], y=X_trans_tsne[:,1],
                            color=y.astype(str),
                            height=600, width=600, title = 'TSNE 2D'
                        )
        fig_tsne.update_traces(marker=dict(size=10))

        fig_pca = px.scatter(None, 
                            x=X_trans_pca[:,0], y=X_trans_pca[:,1],
                            color=y.astype(str),
                            height=600, width=600, title = 'PCA 2D'
                        )

        fig_tsne.update_traces(marker=dict(size=1))


    else:
        print("Dimension can be only 2 or 3")

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Dimension Reduction"),
    html.Div(className='row', children=[
        dcc.Graph(id="isomap", figure = fig_iso, style={'display': 'inline-block', 'width': '40%', 'height': '40%'}),
        dcc.Graph(id="tsne", figure = fig_tsne, style={'display': 'inline-block','width': '40%', 'height': '40%'})
    ]),
    html.Div(className='row', children=[
        dcc.Graph(id="pca", figure = fig_pca, style={'display': 'inline-block', 'width': '40%', 'height': '40%'})
    ])
])


if __name__ == '__main__':
    app.run_server()

