import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context
import dash_bootstrap_components as dbc
import os
from pathlib import Path
import json
import base64
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.utils import convert_files_to_docs
from haystack.nodes import CohereReranker
from dotenv import load_dotenv
load_dotenv()


# --- Config initiale
UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(exist_ok=True)
LOG_PATH = Path("index_log.json")

document_store = InMemoryDocumentStore(embedding_dim=1536, similarity="cosine")
embedder = OpenAIDocumentEmbedder(api_key=os.environ.get("OPENAI_API_KEY"), model="text-embedding-3-small")
writer = DocumentWriter(document_store=document_store)
retriever = InMemoryEmbeddingRetriever(document_store=document_store, embedding_model=embedder)
reranker = CohereReranker(api_key="y7P0AgxIvdVbHoELcbtAn4Osb1qmVMpNZilfXR2P", top_k=3)

# --- App Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Recherche IA Haystack"

# --- Fonctions utilitaires
def get_file_mod_times():
    return {
        f.name: os.path.getmtime(f)
        for f in UPLOAD_DIR.iterdir()
        if f.is_file()
    }

def load_previous_log():
    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    return {}

def save_log(log_data):
    with open(LOG_PATH, "w") as f:
        json.dump(log_data, f)

def get_files_to_index(current_log, previous_log):
    return [
        filename for filename, mod_time in current_log.items()
        if filename not in previous_log or mod_time != previous_log[filename]
    ]

def save_uploaded_file(name, content):
    data = content.encode("utf8").split(b";base64,")[1]
    with open(UPLOAD_DIR / name, "wb") as f:
        f.write(base64.decodebytes(data))

# --- Layout principal
app.layout = dbc.Container([
    html.H1("📄 Recherche intelligente avec Haystack", className="my-4"),

    html.H4("1. 📤 Importer vos documents"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Glissez-déposez ou ',
            html.A('sélectionnez vos fichiers')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
        },
        multiple=True
    ),
    html.Div(id='upload-status', className="my-2"),

    html.Hr(),

    html.H4("2. ⚙️ Indexer les documents"),
    dbc.Button("Lancer l'indexation", id='index-btn', color="primary", className="mb-3"),
    html.Div(id='index-status'),

    html.Hr(),

    html.H4("3. 🔍 Poser une question"),
    dcc.Input(id="user-query", type="text", placeholder="Entrez votre question...", style={'width': '100%'}),
    dbc.Button("Rechercher", id="search-btn", color="info", className="my-2"),
    html.Div(id="search-results")
])

# --- Callback upload
@app.callback(
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def upload_files(contents, filenames):
    if contents is not None:
        for name, content in zip(filenames, contents):
            save_uploaded_file(name, content)
        return dbc.Alert("✅ Fichiers enregistrés. Prêts pour l'indexation.", color="success")
    return ""

# --- Callback indexation
@app.callback(
    Output('index-status', 'children'),
    Input('index-btn', 'n_clicks'),
    prevent_initial_call=True
)
def index_documents(n_clicks):
    current_log = get_file_mod_times()
    previous_log = load_previous_log()
    files_to_index = get_files_to_index(current_log, previous_log)

    if files_to_index:
        docs = convert_files_to_docs(dir_path=UPLOAD_DIR, file_paths=files_to_index)
        embedded_docs = embedder.run(docs)["documents"]
        writer.run(documents=embedded_docs)
        save_log(current_log)
        return dbc.Alert("✅ Indexation mise à jour avec succès.", color="success")
    else:
        return dbc.Alert("✅ Aucun nouveau fichier détecté. Indexation sautée.", color="secondary")

# --- Callback recherche
@app.callback(
    Output('search-results', 'children'),
    Input('search-btn', 'n_clicks'),
    State('user-query', 'value'),
    prevent_initial_call=True
)
def search_documents(n_clicks, query):
    if not query:
        return dbc.Alert("⚠️ Veuillez saisir une question.", color="warning")

    results = retriever.run(query=query, top_k=5)["documents"]
    if not results:
        return dbc.Alert("❌ Aucun résultat trouvé.", color="danger")

    return html.Div([
        html.H5("📌 Résultats les plus pertinents :"),
        *[
            html.Div([
                html.Strong(f"📄 {doc.meta.get('name', 'Document')}"),
                html.P(doc.content[:1000]),
                html.Hr()
            ])
            for doc in results
        ]
    ])

# --- Lancer l'app
if __name__ == '__main__':
    app.run_server(debug=True)
