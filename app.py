import streamlit as st
import cv2
import numpy as np
from PIL import Image

def perform_edge_detection(image, method='Canny', lower_threshold=50, upper_threshold=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'Canny':
        edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    elif method == 'Sobel':
        edges_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        edges_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.sqrt(edges_x**2 + edges_y**2)
        edges = np.uint8(edges)
    else:
        st.error("Invalid edge detection method.")
        return None

    return edges

def perform_corner_detection(image, method='Harris', block_size=2, ksize=3, k=0.04):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'Harris':
        corners = cv2.cornerHarris(gray, block_size, ksize, k)
        corners = cv2.dilate(corners, None)
        image[corners > 0.01 * corners.max()] = [0, 0, 255]  # Highlight corners in red
    elif method == 'Shi-Tomasi':
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            corners = np.int0(corners)
            for i in corners:
                x, y = i.ravel()
                cv2.circle(image, (x, y), 3, 255, -1)
    else:
        st.error("Invalid corner detection method.")
        return None

    return image

def main():
    st.title("Edge and Corner Detection with OpenCV and Streamlit")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Edge Detection
        st.header("Edge Detection")
        edge_method = st.radio("Select edge detection method:", ['Canny', 'Sobel'])
        lower_threshold = st.slider("Lower Threshold", 0, 255, 50)
        upper_threshold = st.slider("Upper Threshold", 0, 255, 150)

        edges = perform_edge_detection(image, method=edge_method, lower_threshold=lower_threshold,
                                       upper_threshold=upper_threshold)
        if edges is not None:
            st.image(edges, caption=f"{edge_method} Edge Detection", use_column_width=True)

        # Corner Detection
        st.header("Corner Detection")
        corner_method = st.radio("Select corner detection method:", ['Harris', 'Shi-Tomasi'])
        block_size = st.slider("Block Size", 2, 10, 2)
        ksize = st.slider("Kernel Size", 3, 7, 3)
        k = st.slider("Harris Corner Parameter (k)", 0.01, 0.1, 0.04, step=0.01)

        corners_image = perform_corner_detection(image.copy(), method=corner_method, block_size=block_size,
                                                 ksize=ksize, k=k)
        if corners_image is not None:
            st.image(corners_image, caption=f"{corner_method} Corner Detection", use_column_width=True)

if __name__ == "__main__":
    main()