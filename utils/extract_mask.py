import pandas as pd
import numpy as np
import cv2
import os


def main(method='otsu'):
    """"Processing images with OpenCV by Simple Image Processing Algorithms"""

    root_folder = '/mnt/data/dataset/apple-weight-dataset/v2'
    dest_folder = '/mnt/data/dataset/apple-weight-dataset/v2-mask'
    
    # apple color boundaries [B, G, R]
    lower = [0, 0, 0]
    upper = [179, 255, 138]

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # Create a list to store the results
    results = []

    # Iterate through the root folder and its subfolders
    for foldername, subfolders, filenames in os.walk(root_folder):
        # Create the corresponding subfolders in the destination folder
        for subfolder in subfolders:
            os.makedirs(os.path.join(dest_folder, os.path.relpath(os.path.join(foldername, subfolder), root_folder)), exist_ok=True)
        
        for filename in filenames:
            if filename.endswith('.png'):
                # Load the image and calculate the white pixels of the mask
                image = cv2.imread(os.path.join(foldername, filename))
                
                if method == 'hsv':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    # Find the colors within the specified boundaries and apply the mask
                    binary = cv2.inRange(image, lower, upper)
                    _, thresh = cv2.threshold(binary, 40, 255, 0)

                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
                    largest_item = sorted_contours[0]
                    area = cv2.contourArea(largest_item)
                    
                    # Obtain binary mask from largest contour
                    mask = np.zeros_like(image[:,:,0])
                    cv2.drawContours(mask, [largest_item], -1, 255, -1)
                
                if method == 'canny':
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    v = np.median(gray)
                    sigma = 0.33
                    
                    # Applying automatic Canny edge detection using the computed median
                    lower = int(max(0, (1.0 - sigma) * v))
                    upper = int(min(255, (1.0 + sigma) * v))
                    edge = cv2.Canny(gray, lower, upper)
                    
                    # Close the edges
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
                    
                    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                
                if method == 'otsu':
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Ostu thresholding
                    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                    
                # Save the mask image in the destination folder with the same structure as the root folder
                mask_filename = os.path.join(dest_folder, os.path.relpath(os.path.join(foldername, filename), root_folder))
                mask_folder = os.path.dirname(mask_filename)
                os.makedirs(mask_folder, exist_ok=True)
                cv2.imwrite(mask_filename, mask)

                # Add the result to the list
                results.append([os.path.join(foldername, filename), area])


    df = pd.DataFrame(results, columns=['image_path', 'area'])
    df['weight'] = df['image_path'].apply(lambda x: int(x.split('/')[-2].split('-')[0]))

    # Define label for based on weight
    bins = [0, 2500, 3500, np.inf]
    labels = [0, 1, 2]

    df['label'] = pd.cut(df['weight'], bins=bins, labels=labels)
    df.to_csv('../assets/v2/dataset_v2.csv', index=False)


if __name__ == '__main__':
    main(method='hsv')