import pandas as pd
import numpy as np
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
import sklearn
import plotly.express as px
import PySimpleGUI as sg
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

sg.theme('DarkBlue2')

layout_file = [[sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key = "-IN-"), sg.Button("Submit")]]

layout_algo = [
    # [[sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key = "-IN-")]],
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

window_file = sg.Window('File', layout_file)
window_algo = sg.Window('Parameters', layout_algo)
while True:
    event_file, values_file = window_file.read()
    if event_file == sg.WIN_CLOSED:
        break
    data = pd.read_csv(values_file["-IN-"])
    filename = str(values_file["-IN-"])
    print(data.head())

    event_algo, values_algo = window_algo.read()
    if event_algo == sg.WIN_CLOSED:
        break
    # data = pd.read_csv(values_algo["-IN-"])
    # print(data.head())
    if values_algo[0] == '':
        values_algo[0] = 5
    neighbors_iso = int(values_algo[0])
    dim_iso = int(values_algo[1])
    if values_algo[2] == '':
        values_algo[2] = 30
    neighbors_tsne = int(values_algo[2])
    dim_tsne = int(values_algo[3])
    if values_algo[4] == '':
        values_algo[4] = 1000
    iter_tsne = int(values_algo[4])
    dim_pca = int(values_algo[5])
    print('Isomap Nearest Neighbors: ', values_algo[0])
    print('Isomap no of dimensions to reduce to: ', values_algo[1])
    print('TSNE Nearest Neighbors: ', values_algo[2])
    print('TSNE no of dimensions to reduce to: ', values_algo[3])
    print('TSNE no of iterations: ', values_algo[4])
    print('PCA no of dimensions to reduce to: ', values_algo[5])

    X = data[data.columns[:-1]]
    y = data[data.columns[-1]]
    print("Shape of the matrix with independent variables: ",X.shape)
    print("Shape of the matrix with dependent variable", y.shape)

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

        fig_pca.update_traces(marker=dict(size=3))


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

        fig_pca.update_traces(marker=dict(size=1))


    else:
        print("Dimension can be only 2 or 3")

app = Dash(__name__)

app.layout = html.Div([
    html.Div(children = [
        html.H1("VISUALIZATION OF DIMENSION REDUCTION TECHNIQUES", style={'color':'navy', 'textAlign':'center'}),
        html.H3(filename, style = {'color':'black', 'textAlign': 'center'})
    ]),
    html.Div(className='row', children=[
        dcc.Graph(id="isomap", figure = fig_iso, style={'display': 'inline-block', 'width': '50%', 'height': '50%'}),
        dcc.Graph(id="tsne", figure = fig_tsne, style={'display': 'inline-block','width': '40%', 'height': '40%'})
    ]),
    html.Div(className='row', children=[
        dcc.Graph(id="pca", figure = fig_pca, style={'display': 'inline-block', 'width': '40%', 'height': '40%'})
    ])
])


if __name__ == '__main__':
    app.run_server()

