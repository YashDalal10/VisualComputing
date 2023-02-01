import pandas as pd
import numpy as np
from sklearn.manifold import Isomap, TSNE
import sklearn
import plotly.express as px
from sklearn.datasets import load_digits
import PySimpleGUI as sg
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

X,y = load_digits(return_X_y = True)
X.shape, y.shape

input_file = [[sg.Text("Choose a file: "), sg.Input(), ]]

layout_iso = [
    [sg.Text('Isomap algorithm requires follwoing parameters - ')],
    [sg.Text('Number of neighbors (default=5)'), sg.InputText()],
    # [sg.Text('radius (default=None) (helpful for limiting the distance')],
    [sg.Text('Number of dimensions (can take only 2 or 3)'), sg.InputText()],
    # [sg.Text('eigen_solver (default=auto) (arpack - Arnoldi decomposition or dense - LAPACK)')],
    # [sg.Text('Path method (default=auto)(FW - Floyd-Warshall or D - Dijkstras)')],
    # [sg.Text('Neighbors algorithm (default=auto)(brute, kd tree and ball tree)')],
    # [sg.Text('n_jobs (default=-1, all processors)')],
    # [sg.Text('n_iter'), sg.InputText()],
    [sg.Button('Reduce and Plot')]
]

layout_tsne = [
    [sg.Text('TSNE algorithm requires follwoing parameters - ')],
    [sg.Text('Perplexity (no of neighbors) (default=30.0)'), sg.InputText()],
    # [sg.Text('radius (default=None) (helpful for limiting the distance')],
    [sg.Text('Number of dimensions (can take only 2 or 3)'), sg.InputText()],
    [sg.Text('Number of iterations (default 1000)'), sg.InputText()],
    # [sg.Text('eigen_solver (default=auto) (arpack - Arnoldi decomposition or dense - LAPACK)')],
    # [sg.Text('Path method (default=auto)(FW - Floyd-Warshall or D - Dijkstras)')],
    # [sg.Text('Neighbors algorithm (default=auto)(brute, kd tree and ball tree)')],
    # [sg.Text('n_jobs (default=-1, all processors)')],
    # [sg.Text('n_iter'), sg.InputText()],
    [sg.Button('Reduce and Plot')]
]

window_iso = sg.Window('Parameters', layout_iso)
window_tsne = sg.Window('Parameters', layout_tsne)

while True:
    event_iso, values_iso = window_iso.read()
    if event_iso == sg.WIN_CLOSED:
        break
    print('Isomap Nearest Neighbors: ', values_iso[0])
    print('Isomap no of dimensions to reduce to: ', values_iso[1])
    neighbors_iso = int(values_iso[0])
    dim_iso = int(values_iso[1])

    event_tsne, values_tsne = window_tsne.read()
    if event_tsne == sg.WIN_CLOSED:
        break
    print('TSNE Nearest Neighbors: ', values_tsne[0])
    print('TSNE no of dimensions to reduce to: ', values_tsne[1])
    print('TSNE iterations: ', values_tsne[2])
    neighbors_tsne = int(values_tsne[0])
    dim_tsne = int(values_tsne[1])
    iter_tsne = int(values_tsne[2])

    if dim_iso != dim_tsne:
        print('Error')

    iso = Isomap(n_neighbors = neighbors_iso, n_components = dim_iso)
    X_trans_iso = iso.fit_transform(X)

    tsne = TSNE(n_components = dim_tsne, perplexity = neighbors_tsne, n_iter = iter_tsne)
    X_trans_tsne = tsne.fit_transform(X)

    if dim_iso == 3 and dim_tsne == 3:
    # Create a 3D scatter plot
        fig_iso = px.scatter_3d(None, 
                            x=X_trans_iso[:,0], y=X_trans_iso[:,1], z=X_trans_iso[:,2],
                            color=y.astype(str),
                            height=700, width=700, title = 'Isomap 3D'
                        )

        # Update chart looks
        # fig_iso.update_layout(#title_text="Scatter 3D Plot",
        #                 showlegend=True,
        #                 legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
        #                 scene_camera=dict(up=dict(x=0, y=0, z=1), 
        #                                         center=dict(x=0, y=0, z=-0.2),
        #                                         eye=dict(x=-1.5, y=1.5, z=0.5)),
        #                                         margin=dict(l=0, r=0, b=0, t=0),
        #                 scene = dict(xaxis=dict(backgroundcolor='white',
        #                                         color='black',
        #                                         gridcolor='#f0f0f0',
        #                                         title_font=dict(size=10),
        #                                         tickfont=dict(size=10),
        #                                         ),
        #                             yaxis=dict(backgroundcolor='white',
        #                                         color='black',
        #                                         gridcolor='#f0f0f0',
        #                                         title_font=dict(size=10),
        #                                         tickfont=dict(size=10),
        #                                         ),
        #                             zaxis=dict(backgroundcolor='lightgrey',
        #                                         color='black', 
        #                                         gridcolor='#f0f0f0',
        #                                         title_font=dict(size=10),
        #                                         tickfont=dict(size=10),
        #                                         )))

        # Update marker size
        fig_iso.update_traces(marker=dict(size=3))

        fig_tsne = px.scatter_3d(None, 
                            x=X_trans_tsne[:,0], y=X_trans_tsne[:,1], z=X_trans_tsne[:,2],
                            color=y.astype(str),
                            height=700, width=700, title = 'TSNE 3D'
                        )

        
        # Update marker size
        fig_tsne.update_traces(marker=dict(size=3))

        # fig.show()
        # fig.write_html("tsne3.html")

    elif dim_iso == 2 and dim_tsne == 2:
        fig_iso = px.scatter(None, 
                            x=X_trans_iso[:,0], y=X_trans_iso[:,1],
                            color=y.astype(str),
                            height=600, width=600, title = 'Isomap 2D'
                        )

        
        fig_iso.update_traces(marker=dict(size=5))

        # fig.show()
        # fig.write_html("isomap2.html")

        fig_tsne = px.scatter(None, 
                            x=X_trans_tsne[:,0], y=X_trans_tsne[:,1],
                            color=y.astype(str),
                            height=600, width=600, title = 'TSNE 2D'
                        )


        # Update marker size
        fig_tsne.update_traces(marker=dict(size=5))

        # fig.show()
        # fig.write_html("tsne3.html")

    else:
        print("Dimension can be only 2 or 3")

app = Dash(__name__)

# app.layout = html.Div(className = "row", children = [
#     html.Div([
#         html.Div(dcc.Graph(id="isomap", figure = fig_iso, style={'display': 'inline-block', 'width': '30%', 'height': '30%'})),
#         html.Div(dcc.Graph(id="tsne", figure = fig_tsne, style={'display': 'inline-block','width': '30%', 'height': '30%'}))
#     ])
# ])
app.layout = html.Div([
    html.H1("Dimension Reduction"),
    # html.H3("Input for Isomap"),
    # dcc.Dropdown(id = 'iso_neighbors', options = [{'label': v, 'value': v} for v in [5,7,9,11]], value = []),
    # dcc.Dropdown(id = 'iso_dim', options = [{'label': v, 'value': v} for v in [2,3]], value = []),
    html.Div(className='row', children=[
        dcc.Graph(id="isomap", figure = fig_iso, style={'display': 'inline-block', 'width': '40%', 'height': '40%'}),
        dcc.Graph(id="tsne", figure = fig_tsne, style={'display': 'inline-block','width': '40%', 'height': '40%'})
    ])
])


if __name__ == '__main__':
    app.run_server()

