import urllib.request as request
import streamlit as st
import streamlit.components.v1 as components

def read_file_contents(file_name):
    ''''
    Read the contents of a file.

    Params:
    ------
    file_name: str
        Path to file.

    Returns:
    -------
    str
    '''
    with open(file_name) as f:
        return f.read()
    

def read_file_contents_web(path):
    """
    Download the content of a file from the GitHub Repo and return as a utf-8 string

    Notes:
    -------
        adapted from 'https://github.com/streamlit/demo-self-driving'

    Parameters:
    ----------
    path: str
        e.g. file_name.md

    Returns:
    --------
    utf-8 str

    """
    response = request.urlopen(path)
    return response.read().decode("utf-8")

def add_logo():
    '''
    Add a logo at the top of the page navigation sidebar

    Approach written by blackary on
    https://discuss.streamlit.io/t/put-logo-and-title-above-on-top-of-page-navigation-in-sidebar-of-multipage-app/28213/5
    
    '''
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://raw.githubusercontent.com/hsma-programme/Teaching_DES_Concepts_Streamlit/main/resources/hsma_logo_transparent_background_small.png);
                background-repeat: no-repeat;
                padding-top: 175px;
                background-position: 40px 30px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Community DES Model";
                padding-left: 20px;
                margin-top: 50px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )

# From https://discuss.streamlit.io/t/st-markdown-does-not-render-mermaid-graphs/25576/3
def mermaid(code: str, height=600) -> None:
    components.html(
        f"""
    <link href='http://fonts.googleapis.com/css?family=Lexend' rel='stylesheet' type='text/css'>

        <pre class="mermaid">
            {code}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=height
    )
