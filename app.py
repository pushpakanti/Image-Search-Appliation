import streamlit as st
import sys
import time
import json
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import zipfile
from datetime import datetime
from pathlib import Path
from src.inference import YOLOv11Inference
from src.utils import save_metadata, load_metadata, get_unique_classes_counts

sys.path.append(str(Path(__file__).parent))

def img_to_base64(image : Image.Image)-> str:
    buffered= io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def init_session_state():
    session_defaults={
        "metadata":None,
        "unique_classes": [],
        "search_results": [],
        "count_options": {},
        "search_params":{
            "search_mode":"Any of selected classes (OR)",
            "selected_classes": [],
            "thresholds": {}
        },
         "show_boxes": True,
         "grid_columns": 3,
         "highlight_matches": True
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
            image_dir= st.text_input("Image directory path: ", placeholder="enter the path")
        with col2:
            model_path= st.text_input("Model weights path: ", "yolo11m.pt")

        if st.button("Start Inference"):
            if image_dir:
                try:
                    with st.spinner("Running object detection..."):
                        inferencer=YOLOv11Inference(model_path)
                        metadata= inferencer.process_directory(image_dir)
                        metadata_path= save_metadata(metadata, image_dir)                   
                        st.success(f"Processed all images. Metadata saved ")
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
                        st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                        st.success(f" Successfully loaded Metadata for given images..")
                except Exception as e:
                    st.error(f"Error loading metadata: {str(e)}")
        else:
            st.warning(f"Please enter a metadata file path")

if st.session_state.metadata:
    st.header("Search Engine")

    with st.container():
        st.session_state.search_params["search_mode"]= st.radio("Search mode:",
                 ("Any of selected classes (OR)", "All selected classes(AND)"),
                 horizontal=True
        )

        st.session_state.search_params["selected_classes"]= st.multiselect(
            "Classes to search for:",
            options= st.session_state.unique_classes
        )

        if st.session_state.search_params["selected_classes"]:
            st.subheader("Count Thresholds (optional)")
            cols= st.columns(len(st.session_state.search_params["selected_classes"]))
            for i, cls in enumerate(st.session_state.search_params["selected_classes"]):
                with cols[i]:
                    st.session_state.search_params["thresholds"][cls]= st.selectbox(
                        f"Max count for {cls}",
                        options=["None"]+st.session_state.count_options[cls]  
                    )

        if st.button("Search Images", type="primary") and st.session_state.search_params["selected_classes"]:
            results=[]
            search_params= st.session_state.search_params

            for item in st.session_state.metadata:
                matches= False
                class_matches= {}

                for cls in search_params["selected_classes"]:
                    class_detections= [d for d in item['detections'] if d['class']== cls]
                    class_count= len(class_detections)
                    class_matches[cls]= False

                    threshold= search_params['thresholds'].get(cls, "None")
                    if threshold=="None":
                        class_matches[cls]= (class_count>=1)
                    else:
                        class_matches[cls]= (class_count>=1 and class_count<= int(threshold))

                if search_params["search_mode"]=="Any of selected classes (OR)":
                    matches = any(class_matches.values())
                else:
                    matches = all(class_matches.values())


                if matches:
                    results.append(item)

            st.session_state.search_results= results


# Displaying result
if st.session_state.search_results:
    results= st.session_state.search_results
    search_params= st.session_state.search_params

    st.subheader(f" Results: {len(results)} matching images")

    #Display controls
    with st.expander("Display Options: ", expanded= True):
        cols= st.columns(3)
        with cols[0]:
            st.session_state.show_boxes= st.checkbox(
                "Show bounding boxes",
                value= st.session_state.show_boxes 
            )
        with cols[1]:
            st.session_state.grid_columns= st.slider(
                "Grid Columns",
                min_value=2,
                max_value=6,
                value=st.session_state.grid_columns
            )
        with cols[2]:
            st.session_state.highlight_matches= st.checkbox(
                "Highlight matching classes",
                value= st.session_state.highlight_matches

            )

# creating grid
    grid_cols= st.columns(st.session_state.grid_columns)
    col_index=0

    for result in results:
        with grid_cols[col_index]:
                try:
                    img= Image.open(result["image_path"])
                    draw=ImageDraw.Draw(img)

                    if st.session_state.show_boxes:
                        try:
                            font= ImageFont.truetype("arial.ttf", 12)
                        except:
                            font= ImageFont.load_default()
                        for det in result ['detections']:
                            cls= det['class']
                            bbox= det['bbox']

                            if cls in search_params["selected_classes"]:
                                color= "#1222D0"
                                thickness= 2
                            elif not st.session_state.highlight_matches:
                                color="#579612"
                                thickness= 1
                            else:
                                continue

                            draw.rectangle(bbox, outline=color, width=thickness)

                            if cls in search_params["selected_classes"] or not st.session_state.highlight_matches:
                                label= f"{cls} {det['confidence']:.2f}"
                                text_bbox= draw.textbbox((0,0), label, font=font)
                                text_width= text_bbox[2] - text_bbox[0]
                                text_height= text_bbox[3] - text_bbox[1]

                                draw.rectangle(
                                    [bbox[0], bbox[1], bbox[0] + text_width + 8, bbox[1] + text_height + 4],
                                    fill=color
                                )

                            
                                draw.text(
                                    (bbox[0]+4, bbox[1]+2),
                                    label,
                                    fill='white',
                                    font=font
                                )

                        meta_items = [f"{k}: {v}" for k, v in result["class_counts"].items()
                                        if k in search_params["selected_classes"]]
                    

                        # Display card
                        st.markdown(f"""
                        <div class="image-card">
                            <div class="image-container">
                                <img src="data:image/png;base64,{img_to_base64(img)}">
                            </div>
                            <div class="meta-overlay">
                                <strong>{Path(result['image_path']).name}</strong><br>
                                {", ".join(meta_items) if meta_items else "No matches"}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)   
                             
                except Exception as e:
                    st.error(f"Error displaying {result['image_path']}: {str(e)}")
                
        col_index= (col_index+1) % st.session_state.grid_columns

    with st.expander("Export Options"):
        st.download_button(
            label= "Download Results (JSON)",
            data= json.dumps(results, indent=2),
            file_name="search_results.json",
            mime="application/json"
        )

        # Option to include images in a zip
        if st.button("Prepare Images ZIP"):
            try:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for result in results:
                        img_path = result["image_path"]
                        img_name = Path(img_path).name
                        zipf.write(img_path, arcname=img_name)
                zip_buffer.seek(0)

                st.session_state.zip_data = zip_buffer.getvalue()
                st.success("ZIP file prepared successfully! You can now download it below.")
            except Exception as e:
                st.error(f"Error preparing ZIP: {str(e)}")

        # if ZIP is ready
        if "zip_data" in st.session_state:
            st.download_button(
                label="Download All Images (ZIP)",
                data=st.session_state.zip_data,
                file_name="search_results_images.zip",
                mime="application/zip"
            )
