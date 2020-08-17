# OJOS human tracking and replacement tool
This tool allows replacing humans and moving object in a video with stick figures or other images. This improves the privacy of the clients as the video can be saved without their picture appearing and keep their identity safe even in case of breaches.
## Tools involved:
1. Use HOG + SVM detector provided by OpenCV2 to detect bounding boxes
2. Use Non-maxima suppression to combine overlapping detections
## TODO
1. Use trackers to speed up
2. Use Human segmentation model (neural network) to improve accuracy and get specific pixels to replace
3. Customize stick figure pose to match (as much as possible) the pose of the client --not a priority--