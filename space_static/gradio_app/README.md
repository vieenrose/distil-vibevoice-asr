# Live Gradio app (needs GPU hardware)

Ready-to-run Gradio app implementing the same chunked pipeline live
(upload audio -> streaming transcript). To use it: create a Space with the
Gradio SDK on ZeroGPU or dedicated GPU hardware (requires a paid HF plan),
copy `app.py` + `requirements.txt` to its root, and copy the `examples/`
folder from this Space for the precomputed long clips.
