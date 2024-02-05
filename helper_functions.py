import streamlit.components.v1 as components

# def d2(code: str) -> None:
#     components.html(
#         f"""
#         <pre class="d2">
#             {code}
#         </pre>

#         <script type="module">
#             import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
#             mermaid.initialize({{ startOnLoad: true }});
#         </script>
#         """
#     )

# From https://discuss.streamlit.io/t/st-markdown-does-not-render-mermaid-graphs/25576/3
def mermaid(code: str, height=600, width=None) -> None:
    components.html(
        f"""


        <pre class="mermaid">
            {code}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=height,
        width=width
    )
