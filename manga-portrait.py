import cv2
import numpy as np

# This code is major used for portrait photos.

class Cartoonizer:
    """Cartoonizer effect
        A class that applies a cartoon effect to an image.
        The class uses a bilateral filter and adaptive thresholding to create
        a cartoon effect.
    """
    def __init__(self):
        pass

    def render(self, img_rgb):
        img_rgb = cv2.imread(img_rgb)   

        numDownSamples = 2       # number of downscaling steps
        numBilateralFilters = 50  # number of bilateral filtering steps

        # -- STEP 1 --
        # downsample image using Gaussian pyramid
        img_color = img_rgb
        # show original image using imshow
        cv2.imshow("original", img_color)
        cv2.waitKey(100)
        
        for _ in xrange(numDownSamples):
            img_color = cv2.pyrDown(img_color)
        # show resized image using imshow
        cv2.imshow("downsampled", img_color)
        cv2.waitKey(100)
        
        # enhance contrast
        img_color = self.enhanceContrast(img_color)
        
        # repeatedly apply small bilateral filter instead of applying
        # one large filter
        for _ in xrange(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        
        # show filtered image using imshow
        cv2.imshow("filtered", img_color)
        cv2.waitKey(100)
        # upsample image to original size
        for _ in xrange(numDownSamples):
            img_color = cv2.pyrUp(img_color)
        
        # show upsampled image using imshow
        cv2.imshow("upsampled", img_color)
        cv2.waitKey(100)
        # -- STEPS 2 and 3 --
        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)
        # show the grey image using imshow
        cv2.imshow("grey", img_blur)
        cv2.waitKey(100)
        
        # -- STEP 4 --
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
        # show the edge image using imshow
        cv2.imshow("edge", img_edge)
        cv2.waitKey(100)

        # -- STEP 5 --
        # convert back to color so that it can be bit-ANDed with color image
        (x,y,z) = img_color.shape
        img_edge = cv2.resize(img_edge,(y,x)) 
        # convert back to color
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        cv2.imwrite("/Users/user/Downloads/portrait_edge.png",img_edge)
        # show the step 5 image using imshow
        cv2.imshow("final", img_edge)
        
        blackimg = cv2.imread("/Users/user/Downloads/portrait_edge.png")
        # Level 1 filter
        dst = cv2.fastNlMeansDenoisingColored(blackimg, None, 120, 10, 7, 25)
        # Level 2 filter
        dst = cv2.fastNlMeansDenoisingColored(dst, None, 20, 10, 7, 25)
        # Final clear up
        dst = self.clearPortrait(dst)
        cv2.imwrite("/Users/user/Downloads/portrait_bw.png", dst)
        
        cv2.waitKey(100)
        #img_edge = cv2.resize(img_edge,(i for i in img_color.shape[:2]))
        #print img_edge.shape, img_color.shape
        return cv2.bitwise_and(img_color, dst)
    
    def enhanceContrast(self, img_rgb):
        alpha = 2.0
        beta = 0
        new_image = np.zeros(img_rgb.shape, img_rgb.dtype)
        for y in range(img_rgb.shape[0]):
            for x in range(img_rgb.shape[1]):
                for c in range(img_rgb.shape[2]):
                    new_image[y,x,c] = np.clip(alpha*img_rgb[y,x,c] + beta, 0, 255)
        cv2.imshow("Original Image", img_rgb)
        cv2.waitKey(100)
        cv2.imshow("New Image", new_image)
        cv2.waitKey(100)
        return new_image

tmp_canvas = Cartoonizer()
file_name = "/Users/user/Downloads/trump2.png" #File_name will come here
res = tmp_canvas.render(file_name)
cv2.imwrite("/Users/user/Downloads/portrait_cartoon.jpg", res)
cv2.imshow("Cartoon version", res)
cv2.waitKey(100)
cv2.destroyAllWindows()