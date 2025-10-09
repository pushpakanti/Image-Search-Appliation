import streamlit as st
import sys
import time
from pathlib import Path
from src.inference import YOLOv11Inference
from src.utils import save_metadata, load_metadata, get_unique_classes_counts

sys.path.append(str(Path(__file__).parent))

def init_session_state():
    session_defaults={
        "metadata":None,
        "unique_classes": [],
        "count_options": {}
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

st.set_page_config(page_title="YOLOv11 Image Search App", layout="wide")
st.title("Object Search Application using Computer Vision")


##  options--
option= st.radio("Choose an option: ",
                 ("Process new images", "Load existing metadata"),
                 horizontal= True)

if option== "Process new images":
    with st.expander("process new images", expanded=True):
        col1, col2= st.columns(2)
        with col1:
            image_dir= st.text_input("Image directory path: ", placeholder="path/to")
        with col2:
            model_path= st.text_input("Model weights path: ", "yolo11m.pt")

        if st.button("Start Inference"):
            if image_dir:
                try:
                    with st.spinner("Running object detection..."):
                        inferencer=YOLOv11Inference(model_path)
                        metadata= inferencer.process_directory(image_dir)
                        metadata_path= save_metadata(metadata, image_dir)                   
                        st.success(f"Processed {len(metadata)} images. Metadata saved to: ")
                        st.code(str(metadata_path))
                        st.session_state.metadata=metadata
                        st.session_state.unique_classes, st.session_state.count_options= get_unique_classes_counts(metadata)
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
            else:
                st.warning(f"Please enter an image directory path")

else:
    with st.expander("Load Existing Metadata", expanded= True):
        metadata_path= st.text_input("Metadata file path: ", placeholder="path/to/metadata.json")

        if st.button("Load Metadata"):
            if metadata_path:
                try:
                    with st.spinner("Loading Metadata..."):
                        metadata= load_metadata(metadata_path)
                        st.session_state.metadata= metadata
                        st.session_state.unique_classes, st.session_state.count_options
                        st.success(f" Successfully loaded Metadata for {len(metadata)} images..")
                except Exception as e:
                    st.error(f"Error loading metadata: {str(e)}")
        else:
            st.warning(f"Please enter a metadata file path")