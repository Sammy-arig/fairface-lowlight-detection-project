Bias & Fairness in Object Detection using YOLOv8
## References

### FairFace Dataset

📄 FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age
https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf

### Project Report

📄 Read the Full Project Report
https://github.com/Sammy-arig/fairface-lowlight-detection-project/blob/main/object_detection_paper.pdf

Developed a computer vision research project to evaluate fairness and demographic bias in YOLOv8 face/object detection across different racial groups using the FairFace dataset. The project compared baseline detection performance with low-light stress testing to analyze how environmental conditions affect detection accuracy and fairness metrics.

Implemented image preprocessing, low-light simulation, confidence analysis, and demographic-wise performance evaluation using Python and YOLOv8. Measured detection rate, false negative rate, demographic parity gap, and equalized odds across seven demographic groups.

Results showed near-perfect detection under standard conditions, while low-light environments introduced measurable disparities across certain groups, highlighting the importance of fairness auditing in real-world AI deployment.

Tech Stack: Python, YOLOv8, OpenCV, Pandas, NumPy, Matplotlib, Deep Learning, Computer Vision
