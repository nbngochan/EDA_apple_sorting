import pandas as pd
import numpy as np
import cv2
import os


def main(method='canny'):
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
                    # find the colors within the specified boundaries and apply the mask
                    mask = cv2.inRange(image, lower, upper)
                    output = cv2.bitwise_and(image, image, mask=mask)

                    ret, thresh = cv2.threshold(mask, 40, 255, 0)

                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
                    largest_item = sorted_contours[0]
                    output = image.copy()
                    cv2.drawContours(output, largest_item, -1, (255,0,0), 10)

                    # Obtain binary mask from largest contour
                    binary = np.zeros(image.shape, np.uint8)
                    cv2.drawContours(binary, [largest_item], -1, 255, -1)
                    
                    # Calcuate the area of the largest contour
                    area = cv2.contourArea(largest_item)
                
                
                    # Save the mask image in the destination folder with the same structure as the root folder
                    mask_filename = os.path.join(dest_folder, os.path.relpath(os.path.join(foldername, filename), root_folder))
                    mask_folder = os.path.dirname(mask_filename)
                    os.makedirs(mask_folder, exist_ok=True)
                    cv2.imwrite(mask_filename, binary[:,:,0])
                
                if method == 'canny':
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Apply Canny edge detection
                    edges = cv2.Canny(gray, 100, 200)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    closed_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                    
                    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                    
                    mask_filename = os.path.join(dest_folder, os.path.relpath(os.path.join(foldername, filename), root_folder))
                    mask_folder = os.path.dirname(mask_filename)
                    os.makedirs(mask_folder, exist_ok=True)
                    cv2.imwrite(mask_filename, mask)

                # Add the result to the list
                results.append([os.path.join(foldername, filename), area])


    # Create a pandas dataframe to store the results
    df = pd.DataFrame(results, columns=['image_path', 'area'])
    df['weight'] = df['image_path'].apply(lambda x: int(x.split('/')[-2].split('-')[0]))

    # Define label for based on weight
    bins = [0, 2500, 3500, np.inf]
    labels = [0, 1, 2]

    # Create a new column for the label
    df['label'] = pd.cut(df['weight'], bins=bins, labels=labels)

    # Save the dataframe to a CSV file
    df.to_csv('../assets/dataset_v2.csv', index=False)


if __name__ == '__main__':
    main(method='hsv')