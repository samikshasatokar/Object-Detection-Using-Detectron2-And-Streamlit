import cv2
import numpy as np
import streamlit as st

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



def initialization():

    cfg = get_cfg()
    # Force model to operate within CPU, remove if CUDA compatible devices ara available
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # Setting the threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # Initializing the model
    predictor = DefaultPredictor(cfg)

    return cfg, predictor



def inference(predictor, img):
    return predictor(img)



def output_image(cfg, img, outputs):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_img = out.get_image()

    return processed_img



def discriminate(outputs, classes_to_detect):
    
    pred_classes = np.array(outputs['instances'].pred_classes)
    # Take the elements matching *classes_to_detect*
    mask = np.isin(pred_classes, classes_to_detect)
    # Get the indexes
    idx = np.nonzero(mask)

    # Get the current Instance values
    pred_boxes = outputs['instances'].pred_boxes
    pred_classes = outputs['instances'].pred_classes
    pred_masks = outputs['instances'].pred_masks
    scores = outputs['instances'].scores

    # Get them as a dictionary and leave only the desired ones with the indexes
    out_fields = outputs['instances'].get_fields()
    out_fields['pred_boxes'] = pred_boxes[idx]
    out_fields['pred_classes'] = pred_classes[idx]
    out_fields['pred_masks'] = pred_masks[idx]
    out_fields['scores'] = scores[idx]

    return outputs


def main():
    # Initialization
    cfg, predictor = initialization()

    # Streamlit initialization
    st.title("Object Detection using Detectron")
    st.markdown(''' Developed by Facebook AI Research (FAIR), Detectron is a highly efficient object detetction codebase.
                    Built to aid the fleeting implementation of computer vision studies, Detectron encompasses implementaions for the following algorithms -''')
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Fast R-CNN", "", "")
    col2.metric("Faster R-CNN", "", "")
    col3.metric("Mask R-CNN", "", "")
    col4.metric("RetinaNet", "", "")
    col5.metric("R-FCN", "", "")
    col6.metric("RPN", "", "")
    st.image("od.jpg")
    
    # Retrieve image
    uploaded_img = st.sidebar.file_uploader("Choose an image to upload...", type=['jpg', 'jpeg', 'png'])
    x = st.sidebar.button("Detect Objects")
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        # Detection code
        if x:
            outputs = inference(predictor, img)
            out_image = output_image(cfg, img, outputs)
            st.image(out_image, caption='Processed Image', use_column_width=True)        


if __name__ == '__main__':
    main()