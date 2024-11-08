import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Đọc ảnh đầu vào
    image = cv2.imread('Bai7/a-nh-ma-n-hi-nh-2023-07-18-lu-1372-9168-1689662616.png', cv2.IMREAD_GRAYSCALE)

    # 1. Phát hiện cạnh bằng toán tử Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(np.absolute(sobel_combined))

    # 2. Phát hiện cạnh bằng toán tử Prewitt
    prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_x = cv2.filter2D(image, -1, prewitt_kernel_x)
    prewitt_y = cv2.filter2D(image, -1, prewitt_kernel_y)
    prewitt_combined = cv2.magnitude(prewitt_x.astype(float), prewitt_y.astype(float))
    prewitt_combined = np.uint8(np.absolute(prewitt_combined))

    # 3. Phát hiện cạnh bằng toán tử Roberts
    roberts_kernel_x = np.array([[1, 0], [0, -1]], dtype=int)
    roberts_kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)
    roberts_x = cv2.filter2D(image, -1, roberts_kernel_x)
    roberts_y = cv2.filter2D(image, -1, roberts_kernel_y)
    roberts_combined = cv2.magnitude(roberts_x.astype(float), roberts_y.astype(float))
    roberts_combined = np.uint8(np.absolute(roberts_combined))

    # 4. Phát hiện cạnh bằng toán tử Canny
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
    canny_edges = cv2.Canny(blurred_image, 50, 150)

    # 5. Phân đoạn bằng Gaussian và phân ngưỡng
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
    _, threshold_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)

    # Hiển thị tất cả ảnh kết quả với không gian giữa các ảnh rộng hơn
    plt.figure(figsize=(15, 10))

    # Hiển thị kết quả Sobel
    plt.subplot(3, 2, 1)
    plt.title("Ảnh gốc")
    plt.imshow(image, cmap='gray')
    plt.subplot(3, 2, 2)
    plt.title("Ảnh sau khi áp dụng Sobel")
    plt.imshow(sobel_combined, cmap='gray')

    # Hiển thị kết quả Prewitt
    plt.subplot(3, 2, 3)
    plt.title("Ảnh sau khi áp dụng Prewitt")
    plt.imshow(prewitt_combined, cmap='gray')

    # Hiển thị kết quả Roberts
    plt.subplot(3, 2, 4)
    plt.title("Ảnh sau khi áp dụng Roberts")
    plt.imshow(roberts_combined, cmap='gray')

    # Hiển thị kết quả Canny
    plt.subplot(3, 2, 5)
    plt.title("Ảnh sau khi áp dụng Canny")
    plt.imshow(canny_edges, cmap='gray')

    # Hiển thị kết quả Gaussian + Threshold
    plt.subplot(3, 2, 6)
    plt.title("Ảnh sau khi phân đoạn ngưỡng")
    plt.imshow(threshold_image, cmap='gray')

    # Điều chỉnh khoảng cách giữa các ảnh
    plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Điều chỉnh khoảng cách ngang (wspace) và dọc (hspace)

    plt.show()


if __name__ == "__main__":
    main()
