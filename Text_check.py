import cv2

# Simulating a bounding box
rectangle = [100, 150, 400, 300]  # [x1, y1, x2, y2]
dist_text = "Distance: 150.50 cm"  # Example text

# Load a blank image or use an actual image
image = cv2.imread("Pics of guru")

# Displaying the distance text 10 pixels below the bounding box
cv2.putText(image, dist_text, 
            (int(rectangle[0]), int(rectangle[1] + 10)),  # Position of text
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Font and color

# Display the image
cv2.imshow("Image with Distance Text", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
