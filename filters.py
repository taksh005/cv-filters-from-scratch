import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_image(title, image, cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()   

def binary_image(image,threshold):
    binary = np.zeros_like(image)
    binary[image>threshold] = 255
    return binary

def gaussian_mask(size,sigma = 1):
    ax = np.linspace(-(size // 2),size // 2,size)
    xx, yy = np.meshgrid(ax, ax)
    mask = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    return mask/np.sum(mask)

def convolve(image,mask):
    height,width = image.shape
    mask_size = mask.shape[0]
    pad = mask_size // 2
    output = np.zeros((height,width),dtype=np.float32)
    padded_image = np.pad(image,pad,mode = 'constant',constant_values = 0)
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+mask_size,j:j+mask_size]
            output[i,j]=np.sum(region*mask)
    return output

def canny_edge_detection(image,sigma = 1):
    mask = gaussian_mask(5,sigma)
    smoothened = convolve(image,mask)
    magnitude, _, _ = sobel(smoothened)
    return magnitude

def sobel(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = convolve(image,sobel_x)
    grad_y = convolve(image,sobel_y)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    magnitude = (magnitude / magnitude.max()) * 255
    return magnitude.astype(np.uint8), grad_x.astype(np.uint8), grad_y.astype(np.uint8)

def sobel_edge_detection(image, threshold_ratio=0.1):
    magnitude, _, _ = sobel(image)
    threshold = threshold_ratio * np.max(magnitude)  
    return magnitude > threshold

def harris_corner_detection(image,k=0.06,threshold_ratio = 0.01):
    _,grad_x,grad_y = sobel(image)
    I_x2 = grad_x**2
    I_y2 = grad_y**2
    I_xy = grad_x * grad_y
    I_x2  = convolve(I_x2,gaussian_mask(3,sigma = 1))
    I_y2 = convolve(I_y2,gaussian_mask(3,sigma = 1))
    I_xy = convolve(I_xy,gaussian_mask(3,sigma = 1))
    det_M = (I_x2*I_y2)-(I_xy**2)
    trace_M = I_x2 + I_y2
    R = det_M - k*(trace_M**2)
    R[R < 0] = 0 
    threshold = threshold_ratio * R.max()
    return (R > threshold).astype(np.uint8) * 255

def initialization(image, K):
    pixels_x, pixels_y, channels = image.shape
    centroids_x = np.random.randint(0, pixels_x, K)
    centroids_y = np.random.randint(0, pixels_y, K)
    centroids_rgb = image[centroids_x, centroids_y]
    return np.column_stack((centroids_x, centroids_y, centroids_rgb))

def cluster(image, centroid):
    pixels_x,pixels_y,channels = image.shape
    labels = np.zeros((pixels_x,pixels_y),dtype=np.int32)
    for i in range(pixels_x):
        for j in range(pixels_y):
            min_distance = float('inf')
            assigned_cluster = -1

            for k, (centroid_x, centroid_y,r,g,b) in enumerate(centroid):
                spatial_distance = np.sqrt((i - centroid_x) ** 2 + (j - centroid_y) ** 2)
                color_distance = np.sqrt(np.sum((image[i, j] - np.array([r,g,b])) ** 2))
                distance = spatial_distance + color_distance

                if distance < min_distance:
                    min_distance = distance
                    assigned_cluster = k  
            
            labels[i, j] = assigned_cluster  

    return labels

def change_centroids(image,labels,K):
    pixels_x, pixels_y, channels = image.shape
    new_centroids = np.zeros((K, 5),dtype=np.int32)

    for k in range(K):
        cluster_pixels = np.argwhere(labels == k)  
        if len(cluster_pixels) > 0:
           mean_x, mean_y = np.mean(cluster_pixels, axis=0) 
           mean_rgb = np.mean(image[labels == k], axis=0) 
           mean_rgb = np.array(mean_rgb).flatten()  
           new_centroids[k] = np.hstack((mean_x, mean_y, mean_rgb[:3]))
        else:
            rand_x = np.random.randint(0, pixels_x)
            rand_y = np.random.randint(0, pixels_y)
            new_centroids[k] = np.hstack((rand_x, rand_y, image[rand_x, rand_y]))
  
    return new_centroids

def k_means_segmentation(image, K, max_iters, tol):
    centroids = initialization(image, K)
    
    for _ in range(max_iters):
        labels = cluster(image, centroids)
        new_centroids = change_centroids(image, labels, K)

        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    for k in range(K):
        mask = (labels == k)
        if np.any(mask):
            segmented_image[mask] = int(np.mean(image[mask]))

    return segmented_image         

if __name__ == "__main__":
    img_path = 'sample.jpg' 
    image_bgr = cv2.imread(img_path)
    
    if image_bgr is None:
        print(f"Error: Could not load image at '{img_path}'. Please check the path.")
    else:
        # Resize image to a smaller size.
        image_bgr = cv2.resize(image_bgr, (100, 100))
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        show_image("Original Grayscale", image_gray)
        show_image("Original RGB", image_rgb, cmap=None)
        
        # --- 1. Binary Image ---
        print("Processing Binary Image...")
        binary = binary_image(image_gray, threshold=127)
        show_image("Binary Image (Threshold=127)", binary)
        
        # --- 2. Convolution (Gaussian Blur) ---
        print("Processing Convolution (Gaussian Blur)...")
        g_mask = gaussian_mask(size=5, sigma=1)
        blurred = convolve(image_gray, g_mask)
        show_image("Gaussian Blur via Convolution", blurred)
        
        # --- 3. Custom Canny Edge Detection ---
        print("Processing Canny Edge Detection...")
        canny_edges = canny_edge_detection(image_gray, sigma=1)
        show_image("Canny Edge Detection", canny_edges)
        
        # --- 4. Sobel Filter Components ---
        print("Processing Sobel Filter...")
        magnitude, grad_x, grad_y = sobel(image_gray)
        show_image("Sobel Magnitude", magnitude)
        show_image("Sobel Gradient X", grad_x)
        show_image("Sobel Gradient Y", grad_y)
        
        # --- 5. Thresholded Sobel Edge Detection ---
        print("Processing Sobel Edge Detection...")
        sobel_edges = sobel_edge_detection(image_gray, threshold_ratio=0.1)
        show_image("Sobel Edges", sobel_edges.astype(np.uint8) * 255)
        
        # --- 6. Harris Corner Detection ---
        print("Processing Harris Corner Detection...")
        corners = harris_corner_detection(image_gray, k=0.06, threshold_ratio=0.01)
        show_image("Harris Corners", corners)
        
        # --- 7. K-Means Segmentation ---
        print("Processing K-Means Segmentation (this may take a moment)...")
        segmented_img = k_means_segmentation(image_rgb, K=3, max_iters=5, tol=1.0)
        show_image("K-Means Segmentation (K=3)", segmented_img, cmap=None)
        
        print("All processing complete!")
