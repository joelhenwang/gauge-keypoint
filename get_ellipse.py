from ultralytics import YOLOv10
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import ellipse


def get_centered_xy_coords(box):
    xyxy = box.xyxy[0]
    x = xyxy[0] - (xyxy[0] - xyxy[2])/2
    y = xyxy[1] - (xyxy[1] - xyxy[3])/2
    return x.cpu(), y.cpu()

def fit_ellipse(x0, y0, points):
    translated_pts = [(x - x0, y - y0) for x, y in points ]
    D = np.array( [ [x**2, x*y, y**2, x, y, 1] for x, y in translated_pts ] )

    _, _, V = np.linalg.svd(D)
    params = V[-1, :]  # The solution is the last row of V
    
    # Adjust the parameters to account for the translation
    A, B, C, D, E, F = params
    
    # Translate back
    D = D - 2*A*x0 - B*y0
    E = E - 2*C*y0 - B*x0
    F = F + A*x0**2 + C*y0**2 + B*x0*y0 - D*x0 - E*y0

    return A, B, C, D, E, F

def get_ellipse_params(A, B, C, D, E, F):
    # Calculate center of the ellipse
    x0 = (2*C*D - B*E) / (B**2 - 4*A*C)
    y0 = (2*A*E - B*D) / (B**2 - 4*A*C)
    
    # Calculate the semi-major and semi-minor axes
    term1 = 2 * (A*E**2 + C*D**2 + F*B**2 - 2*B*D*E - 4*A*C*F)
    term2 = (B**2 - 4*A*C)
    up = np.sqrt(1 + (4*A*C) / (B**2))
    
    a = np.sqrt(term1 / (term2 * (C*up - (A + C))))
    b = np.sqrt(term1 / (term2 * (A*up - (A + C))))
    
    # Calculate the orientation of the ellipse
    if B == 0:
        if A < C:
            angle = 0
        else:
            angle = 90
    else:
        angle = np.arctan2(B, A - C) / 2
    
    angle = np.degrees(angle)
    
    return x0, y0, a, b, angle




def draw_points_on_image(image, points):
    for pos in points:
        x = int(pos[0].item())
        y = int(pos[1].item())

        image = cv2.drawMarker(
            image, (x, y), color=[255, 0, 0], thickness=4,
            markerType=cv2.MARKER_CROSS, line_type=cv2.LINE_AA,
            markerSize=10
        )

    return image

def draw_ellipse_on_image(image, x0, y0, a, b, angle):
    # Convert (x0, y0) to integer for drawing
    center = (int(x0), int(y0))
    
    # OpenCV uses degrees counterclockwise for angle
    angle = -angle
    
    # Calculate axes
    axes = (int(a), int(b))
    
    # Draw the ellipse
    color = (0, 255, 0)  # Green color
    thickness = 2
    image = cv2.ellipse(image, center, axes, angle, 0, 360, color, thickness)
    
    return image

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


model = YOLOv10("/home/joelwang98/Desktop/yolov10_gauge_start_center_end/best3.pt")
img_path = "/home/joelwang98/Desktop/yolov10_gauge_start_center_end/gauge4.png"

image = cv2.imread(img_path)
image = cv2.resize(image, dsize=[640, 640], interpolation=cv2.INTER_CUBIC)

results = model(image)


for i, result in enumerate(results):
    
    boxes = result.boxes

    class_list = ["center", "end", "middle", "start"]
    
    boxes_xy = boxes.xyxy
    classes = boxes.cls

    middle_boxes = []
    center_box = []
    start_box = []
    end_box = []


    for box in boxes:
        class_idx = int(box.cls[0].item())
        curr_class = class_list[class_idx]
        
        if curr_class == "center":
            center_box.append(box)
            

        if curr_class == "start":
            start_box.append(box)
        
        if curr_class == "end":
            end_box.append(box)

        if curr_class == "middle":
            middle_boxes.append(box)
    
    if len(middle_boxes) < 3:
        raise ValueError("Need to have more than 2 ellipsed refereces in order to fit equation.")
    

    x0, y0 = get_centered_xy_coords(center_box[0])

    points = []

    for box in middle_boxes:
        x, y = get_centered_xy_coords(box)
        points.append((x, y))




    x = []
    y = []
    
    for point in points:
        x.append(point[0].item())
        y.append(point[1].item())
    
    x = np.array(x)
    y = np.array(y)
    coeffs = ellipse.fit_ellipse(x, y)


    X0, Y0, ap, bp, phi = ellipse.cart_to_pol(coeffs)
    x1, y1 = ellipse.get_ellipse_pts((X0, Y0, ap, bp, phi ))


    for i in range(len(x1)):
        
        image = cv2.drawMarker(
            image, (int(x1[i]), int(y1[i])), color=[255, 0, 0], thickness=1,
            markerType=cv2.MARKER_CROSS, line_type=cv2.LINE_AA,
            markerSize=10
        )

    # A, B, C, D, E, F = fit_ellipse(x0, y0, points)
    # X0, Y0, a, b, angle = get_ellipse_params( A, B, C, D, E, F ) 
    # Draw reference points
    # image = draw_points_on_image(image, points) 
    
    # Draw the ellipse on the image
    # image_with_ellipse = draw_ellipse_on_image(image, X0, Y0, a, b, angle)

    cv2.imwrite("fittedEllipse.jpg", image)









