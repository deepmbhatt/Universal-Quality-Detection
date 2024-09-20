# 🏭 **Universal Quality Inspection System** 

![OpenCV](https://img.shields.io/badge/OpenCV-4.5.3-brightgreen)
![YOLO](https://img.shields.io/badge/YOLO-v8-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange)
![Keras](https://img.shields.io/badge/Keras-2.6.0-red)
![Python](https://img.shields.io/badge/Python-3.11.8-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.3.0-lightgrey)

**"INSPECT | DETECT | DEFINE"**

A universal machine learning solution for automating quality inspection and defect detection on manufacturing lines. Say goodbye to slow, inefficient manual inspections and hello to automated accuracy.
![image](https://github.com/user-attachments/assets/09c694d1-c3cf-4583-abc9-3bb92dd739e7)

---

## 🎯 **Problem Statement**

Manual inspections on manufacturing lines are inefficient, slow, and prone to human error. We propose a universal solution for quality inspections using object detection models, capable of detecting defects and classifying objects with precision.

---

## 🚀 **Project Overview**

This project leverages advanced object detection techniques to analyze test data, detect defects, and classify objects. The output is a video highlighting **defective materials with red borders** and **good materials with green borders**.

**Key Technologies**:
- **Object Detection** (YOLO)
- **Machine Learning** (TensorFlow, Keras)
- **Computer Vision** (OpenCV)
- **Data Manipulation** (NumPy, Pandas)

---

## 📂 **Project Flow**

1. **Data Input**: The model accepts training and test data.
2. **Image Processing**: The images are split into two parts:
    - Original Image
    - Binary Image
3. **Object Identification**: Object detection is applied to both original and binary images.
4. **Defect Masking**: Using binary images, objects with defects are selected and masked.
5. **Model Training**: The processed data is used to train the model.
6. **Defect Detection on Test Data**: The trained model detects defects in new test data.
7. **Output Generation**: A video is produced, visually differentiating good (green border) and bad (red border) materials.
![image](https://github.com/user-attachments/assets/b4143837-e465-40cd-b4b8-cfd5213fa5a3)

---

## 🛠 **Technologies Used**

![image](https://github.com/user-attachments/assets/14ce3792-53ef-46e0-9a17-088b1b57855e)


| Technology      | Purpose                               |
|-----------------|---------------------------------------|
| **OpenCV/CV2**  | Image processing and analysis         |
| **YOLO**        | Real-time object detection            |
| **TensorFlow**  | Machine learning model training       |
| **Keras**       | High-level neural networks API        |
| **RoboFlow**    | Data management and annotation        |
| **NumPy**       | Numerical computations                |
| **Pandas**      | Data manipulation and analysis        |
| **Streamlit**   | Interactive web app creation          |

---

## 💡 **Challenges Faced**

- **Data Set Modeling**: Handling large datasets for training and testing.
- **Annotation**: Proper labeling of defect data.
- **Data Noise**: Filtering out noise for more accurate defect detection.
- **Accuracy**: Ensuring high defect detection accuracy.
- **Promptness**: Achieving fast and efficient output generation.
- **ML Pipelines**: Proper machine learning pipelining for training and testing.
- **Defect Classification**: Efficiently classifying defective and good objects.

---

## 📈 **Scalability**

Our model has been designed with scalability in mind. It currently supports universal defect detection for a variety of test data such as:
- **Candles**
- **Cashews**
- **Capsules**
- **Fryums**

With future advancements, the model will be adaptable to any type of object or material, revolutionizing **quality inspection** across industries.

---

## 🔮 **Future Scope**

The next iteration of this project will focus on:
- **Binary-free Defect Detection**: Enhancing the model's ability to detect defects without requiring a binary image.
- **Clustering Algorithms**: Implementing clustering techniques for unsupervised defect detection.
- **Wider Applicability**: Scaling the solution for any form of manufacturing data or product type.

---

## 📜 **Installation Guide**

### Prerequisites:
- Python 3.8+
- OpenCV
- TensorFlow
- Keras
- YOLOv8
- Streamlit

### Steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/universal-quality-inspection.git
   cd universal-quality-inspection
2. Install Dependencies
   ```bash
   pip install -r requirements.txt

3. Run all files for 
   ```bash
   streamlit run app.py

## 📊 Example Use Cases

Manufacturing Lines: Defect detection in production lines for goods like candles, cashews, etc.

Pharmaceuticals: Detecting faulty capsules or products.

Food Industry: Identifying defective fryums and similar items.


## 🤖 Contributing
We welcome contributions from the open-source community! To contribute:

## 🏅 Authors

Your Name - https://github.com/deepmbhatt

Collaborator - [GitHub](https://github.com/deepmbhatt)

## 📧 Contact

For any inquiries, feel free to reach out:

Email: deepmanishbhatt@gmail.com

## ⚖️ License

This project is licensed under the MIT License. See the LICENSE file for more details.

Happy coding! 🎉
